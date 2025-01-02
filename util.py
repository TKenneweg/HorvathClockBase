import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os 
import pickle


class MethylationDataset(Dataset):
    def __init__(self, series_names, data_folder):
        all_dicts = []
        for series_id in series_names:
            series_subfolder = os.path.join(data_folder, series_id)
            pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
            for pkl_file in pkl_files:
                with open(os.path.join(series_subfolder, pkl_file), "rb") as f:
                    sample_dict = pickle.load(f)
                age = sample_dict.get("age", None)
                if age is None or (isinstance(age, float) and np.isnan(age)):
                    continue
                all_dicts.append(sample_dict)

        # Derive probe keys
        all_keys = list(all_dicts[0].keys())
        all_keys.remove("age")

        # Build X, y
        num_samples = len(all_dicts)
        num_probes = len(all_keys)
        print(f"\n[INFO] Building X_data, y_data with {num_samples} samples, {num_probes} probes each.")

        X_data = np.zeros((num_samples, num_probes), dtype=np.float32)
        y_data = np.zeros(num_samples, dtype=np.float32)
        for i, dct in enumerate(all_dicts):
            for j, probe in enumerate(all_keys):
                val = dct[probe]
                X_data[i, j] = val if (val is not None and np.isfinite(val)) else 0.5
            y_data[i] = dct["age"]

        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

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