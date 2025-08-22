import os
import argparse
from src.data_processing import prepare_data
from src.train import training
from src.inference import inference

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on a Small Language Model.")
    parser.add_argument('--action', type=str, required=True, choices=['prepare_data', 'train', 'inference'],
                        help="Action to perform: 'prepare_data', 'train', or 'inference'.")
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help="Prompt for inference.")
    args = parser.parse_args()

    if args.action == 'prepare_data':
        print("Starting data preparation...")
        prepare_data()
        print("Data preparation complete.")

    elif args.action == 'train':
        if not os.path.exists('train.bin') or not os.path.exists('validation.bin'):
            print("Tokenized data not found. Please run --action prepare_data first.")
            return
        print("Starting model training...")
        training()
        print("Training complete.")

    elif args.action == 'inference':
        if not os.path.exists('best_model_params.pt'):
            print("Trained model not found. Please run --action train first.")
            return
        print("Starting inference...")
        inference(args.prompt)

if __name__ == '__main__':
    main()