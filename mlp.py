import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import sys

from util import *
from config import *


def main():
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
    train_size = int(TRAIN_SPLIT_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgePredictorMLP(dataset.X.shape[1], 256, 128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    print("[INFO] Starting training...\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X).squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_mae = total_loss / len(train_loader)

        model.eval()
        total_test_loss = 0
        all_errors = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).squeeze()
                total_test_loss += criterion(preds, batch_y).item()
                all_errors.extend((preds - batch_y).abs().cpu().numpy())
        test_mae = total_test_loss / len(test_loader)
        median_error = float(np.median(all_errors))

        # print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train MAE: {train_mae:.4f}, "
              f"Test MAE: {test_mae:.4f}, "
              f"Test Median Error: {median_error:.4f}")

    torch.save(model, "age_predictor_mlp.pth")
    print("[INFO] Model saved to age_predictor_mlp.pth")

    sys.exit()
    # Histogram of errors
    preds, ages = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds.extend(model(batch_X).squeeze().cpu().numpy())
            ages.extend(batch_y.numpy())
    errors = np.array(preds) - np.array(ages)
    plt.hist(errors, bins=np.arange(-25, 25, 2.5), edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("age_error_histogram.png")
    plt.show()

###############################################################################
# 5. Entry Point
###############################################################################
if __name__ == "__main__":
    main()
