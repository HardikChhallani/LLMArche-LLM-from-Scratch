import os
import numpy as np
from tqdm.auto import tqdm
import tiktoken

def prepare_data():
    """
    Loads the local indian_legal.txt dataset, tokenizes it,
    and saves it to binary files.
    """
    input_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'indian_legal.txt')

    # Read the dataset
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # Split the data into training and validation sets
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Initialize the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile('train.bin')
    val_ids.tofile('validation.bin')

    print("Data preparation complete. train.bin and validation.bin created.")

if __name__ == '__main__':
    prepare_data()