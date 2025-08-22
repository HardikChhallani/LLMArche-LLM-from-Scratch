import torch
import numpy as np
from contextlib import nullcontext
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from src.model import GPT
from src.config import model_config, learning_rate, max_iters, warmup_steps, min_lr, eval_iters, batch_size, block_size, gradient_accumulation_steps

# --- Data Loading ---
def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

# --- Loss Estimation ---
def estimate_loss(model, ctx):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# --- Training Function ---
def training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.set_default_device(device)
    torch.manual_seed(42)

    model = GPT(model_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    best_val_loss = float('inf')
    best_model_params_path = "best_model_params.pt"
    train_loss_list, validation_loss_list = [], []

    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            losses = estimate_loss(model, ctx)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)

        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # Plotting the loss
    train_loss_cpu = [loss.cpu().item() for loss in train_loss_list]
    val_loss_cpu = [loss.cpu().item() for loss in validation_loss_list]

    plt.plot(train_loss_cpu, 'g', label='train_loss')
    plt.plot(val_loss_cpu, 'r', label='validation_loss')
    plt.xlabel("Steps - Every 100 epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    training()