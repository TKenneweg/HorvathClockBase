import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os 
import pickle
from config import *

class MethylationDataset(Dataset):
    def __init__(self, series_names, data_folder):
        nsamples =0
        for series_id in series_names:
            series_subfolder = data_folder + "/" + series_id
            pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
            nsamples += len(pkl_files)

        X_data = np.zeros((nsamples, NUM_PROBES), dtype=np.float32)
        y_data = np.zeros(nsamples, dtype=np.float32)
        print(f"\n[INFO] Building X_data, y_data with {nsamples} samples, {NUM_PROBES} probes each.")

        i = 0
        for series_id in series_names:
            series_subfolder = data_folder + "/" + series_id
            pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
            for pkl_file in pkl_files:
                with open(os.path.join(series_subfolder, pkl_file), "rb") as f:
                    sample_dict = pickle.load(f)
                    X_data[i,:] = list(sample_dict.values())[:-1]
                    y_data[i] = sample_dict["age"]
                    i += 1
            print(f"Loaded {len(pkl_files)} samples from {series_id}")
            

            
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

        # print(self.X.shape, self.y.shape)
        # num_nans = torch.sum(torch.isnan(self.X))
        # invalid_entries = self.X[(self.X > 1) | (self.X < 0)]
        # print(f"Invalid entries in X (greater than 1 or smaller than 0): {invalid_entries}")
        # print(f"Number of NaN entries in X: {num_nans}")
        # print(self.X,self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AgePredictorMLP(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        return self.fc3(x)