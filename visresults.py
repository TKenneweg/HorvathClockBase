import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt


from mlp import *
from config import *


torch.manual_seed(42)
device = "cuda"



model = torch.load("age_predictor_mlp.pth", weights_only=False)
dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
train_size = int(TRAIN_SPLIT_RATIO * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Test size: {test_size}")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Histogram of errors
preds, ages = [], []
model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        preds.extend(model(batch_X).squeeze().cpu().numpy())
        ages.extend(batch_y.numpy())
errors = np.array(preds) - np.array(ages)
# print(errors)
average_error = np.mean(abs(errors))
median_error = np.median(abs(errors))
print("Average error:", average_error)
print("Median error:", median_error)



plt.figure()
plt.scatter(ages, preds, alpha=0.5)
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age")
plt.tight_layout()
plt.savefig("age_scatter_plot.png")
plt.show()


plt.hist(errors, bins=np.arange(-25, 25, 2.5), edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.title("Histogram of Prediction Errors")
plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_error_histogram.png")
plt.show()
