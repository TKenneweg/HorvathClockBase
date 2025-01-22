import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import sys

from config import *


#The Plan 
#Create a pytorch dataset
#Create a 3 layer mlp
#create optimizers and loss function 
#run a standard training loop
#visualize the results

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
                with open(series_subfolder + "/" + pkl_file, "rb") as f:
                    sample_dict = pickle.load(f)
                    X_data[i,:] = list(sample_dict.values())[:-1]
                    y_data[i] = sample_dict["age"]
                    i += 1
            print(f"Loaded {len(pkl_files)} samples from {series_id}")
            

            
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)


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
    
def main():
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
    train_size = int(TRAIN_SPLIT_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    np.save("train_indices.npy", train_dataset.indices)
    np.save("test_indices.npy", test_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgePredictorMLP(dataset.X.shape[1], 256, 128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.L1Loss() #absolute error

    print("[INFO] Starting training...")
    test_errors = [] #for plotting
    test_median_errors = [] #for plotting
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            loss = criterion(model(batch_X).squeeze(), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_X.shape[0]
        scheduler.step()
        train_error = total_train_loss / len(train_dataset)

        model.eval()
        total_test_loss = 0
        all_errors,ages, predicted_ages = [],[],[]
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).squeeze()

                total_test_loss += criterion(preds, batch_y).item() * batch_X.shape[0]
                all_errors.extend((preds - batch_y).abs().cpu().numpy())
                predicted_ages.extend(preds.cpu().numpy())
                ages.extend(batch_y.cpu().numpy())

        test_error = total_test_loss / len(test_dataset)
        test_median_error = float(np.median(all_errors))

        test_errors.append(test_error)
        test_median_errors.append(test_median_error)

        print(f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train Error: {train_error:.4f}, "
              f"Test Error: {test_error:.4f}, "
              f"Test Median Error: {test_median_error:.4f}")

    torch.save(model, "age_predictor_mlp.pth")
    print("[INFO] Model saved to age_predictor_mlp.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), test_errors, label='Test MAE')
    plt.plot(range(1, NUM_EPOCHS + 1), test_median_errors, label='Test Median Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Test Error (Absolute) and Median Error over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_metrics.png")
    plt.show()


    plt.figure()
    plt.scatter(ages, predicted_ages, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.tight_layout()
    plt.savefig("age_scatter_plot.png")
    plt.show()


###############################################################################
# 5. Entry Point
###############################################################################
if __name__ == "__main__":
    main()
