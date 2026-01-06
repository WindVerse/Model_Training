import torch
from torch.utils.data import DataLoader
import os

import config as cfg
from dataset_helpers.dataset import FlagWindDataset
from train_loop import trainModel

if __name__ == "__main__":
    print("Training started...")
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if os.path.exists(cfg.FLAG_DIR) and os.path.exists(cfg.WIND_DIR) and os.path.exists(cfg.TARGET_DIR):
        print("All dataset directories found.")
        print(f"Contents check: {len(os.listdir(cfg.FLAG_DIR))} files found in 'flags'.")
        print(f"Contents check: {len(os.listdir(cfg.WIND_DIR))} files found in 'winds'.")
        print(f"Contents check: {len(os.listdir(cfg.TARGET_DIR))} files found in 'targets'.")
    else:
        print("Error: One or more dataset directories are missing.")
    
    train_set, test_set = FlagWindDataset.load_and_split(train_ratio=0.8)
    
    model = trainModel(train_set, device)