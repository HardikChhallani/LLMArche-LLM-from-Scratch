import torch
import tiktoken
from src.model import GPT
from src.config import model_config

def inference(prompt="Once upon a time"):
    """
    Runs inference on the trained model with a given prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(model_config)
    model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode_ordinary(prompt), device=device).unsqueeze(dim=0)

    generated_tokens = model.generate(context, 200)
    generated_text = enc.decode(generated_tokens.squeeze().tolist())
    print(generated_text)

if __name__ == '__main__':
    inference()