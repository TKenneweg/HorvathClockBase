import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

###############################################################################
# 1. User Configuration
###############################################################################
SERIES_NAMES = ["GSE41037","GSE15745",]

DATA_FOLDER = "./data"

# Hyperparameters
BATCH_SIZE = 32
LR = 2e-4
NUM_EPOCHS = 100
HIDDEN1 = 256
HIDDEN2 = 128
TRAIN_SPLIT_RATIO = 0.8

###############################################################################
# 2. Custom PyTorch Dataset
###############################################################################
class MethylationDataset(Dataset):
    """
    Holds all methylation data in memory.
    X: (num_samples x num_probes)  (torch.FloatTensor)
    y: (num_samples)               (torch.FloatTensor) - ages
    """
    def __init__(self, X, y):
        # Before converting to torch tensors, let's do some debug prints
        print("[DEBUG] MethylationDataset __init__")
        print(f"  X.shape = {X.shape}, y.shape = {y.shape}")
        print(f"  X dtype = {X.dtype}, y dtype = {y.dtype}")
        # Check for NaNs in X, y
        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(y).sum()
        print(f"  X has {x_nan_count} NaN values, y has {y_nan_count} NaN values")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################################################
# 3. Model Definition: 3-layer MLP
###############################################################################
class AgePredictorMLP(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128):
        super(AgePredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)  # single output: predicted age

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################################
# 4. Main Function
###############################################################################
def main():
    print("=== STARTING SCRIPT ===")

    all_dicts = []       # each entry: a dictionary from one GSM
    valid_samples = []   # (series_id, pkl_file_name) or similar

    ############################################################################
    # 4.1 Gather all .pkl files and unify them into an X matrix and y vector
    ############################################################################
    for series_id in SERIES_NAMES:
        series_subfolder = os.path.join(DATA_FOLDER, series_id)
        print(f"\n[INFO] Checking folder: {series_subfolder}")
        if not os.path.isdir(series_subfolder):
            print(f"  [WARNING] Folder does not exist. Skipping {series_id}.")
            continue

        pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
        print(f"  Found {len(pkl_files)} pickle files in {series_subfolder}")

        for pkl_file in pkl_files:
            pkl_path = os.path.join(series_subfolder, pkl_file)

            # Load the dictionary
            with open(pkl_path, "rb") as f:
                sample_dict = pickle.load(f)

            # Check special key "age"
            age = sample_dict.get("age", None)
            if age is None or isinstance(age, float) and np.isnan(age):
                print(f"    [DEBUG] Age is missing or NaN in {pkl_file}, skipping sample.")
                continue

            # Optional: check how many NaNs might be in the data (besides age)
            values = list(sample_dict.values())
            if any(val is None for val in values if isinstance(val, float)):
                print(f"    [DEBUG] Some probe values are None in {pkl_file}, skipping sample.")
                continue

            all_dicts.append(sample_dict)
            valid_samples.append((series_id, pkl_file))

    if not all_dicts:
        print("\n[ERROR] No valid samples with age found. Exiting.")
        return

    # We assume all dictionaries share the same keys in the same order.
    # Let's take the first dictionary's keys as the reference.
    all_keys = list(all_dicts[0].keys())

    # Remove the special key 'age' from the features
    if "age" in all_keys:
        all_keys.remove("age")

    # Check key consistency
    for i, dct in enumerate(all_dicts[1:], start=1):
        d_keys = list(dct.keys())
        if "age" in d_keys:
            d_keys.remove("age")
        if d_keys != all_keys:
            print(f"  [WARNING] Mismatch in keys between dict[0] and dict[{i}].")
            print("  This might cause shape issues or NaNs!")
            # For now, just print and proceed
            break

    num_probes = len(all_keys)
    num_samples = len(all_dicts)
    print(f"\n[INFO] Building X_data, y_data with {num_samples} samples, {num_probes} probes each.")

    # Initialize arrays
    X_data = np.zeros((num_samples, num_probes), dtype=np.float32)
    y_data = np.zeros(num_samples, dtype=np.float32)

    # Fill in the data
    for i, sample_dict in enumerate(all_dicts):
        nan_count = 0
        for j, probe_id in enumerate(all_keys):
            val = sample_dict[probe_id]
            if val is None or not np.isfinite(val):
                nan_count += 1
                val = 0.5
            X_data[i, j] = val
        y_data[i] = sample_dict["age"]

    # Debug prints about X_data, y_data
    print("[DEBUG] X_data shape:", X_data.shape)
    print("[DEBUG] y_data shape:", y_data.shape)
    print("[DEBUG] X_data range: min = {}, max = {}".format(np.nanmin(X_data), np.nanmax(X_data)))
    print("[DEBUG] y_data range: min = {}, max = {}".format(np.nanmin(y_data), np.nanmax(y_data)))
    print("[DEBUG] # of NaNs in X_data =", np.isnan(X_data).sum())
    print("[DEBUG] # of NaNs in y_data =", np.isnan(y_data).sum())

    ############################################################################
    # 4.2 Create PyTorch Dataset and DataLoader
    ############################################################################
    full_dataset = MethylationDataset(X_data, y_data)

    # 80/20 train/test split
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size
    print(f"[INFO] Splitting dataset into train({train_size}) / test({test_size}).")
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[DEBUG] Created DataLoaders.")
    print("  train_loader batches:", len(train_loader))
    print("  test_loader batches:", len(test_loader))

    ############################################################################
    # 4.3 Define Model, Optimizer, Loss
    ############################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = AgePredictorMLP(input_size=num_probes, hidden1=HIDDEN1, hidden2=HIDDEN2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    # Or MSELoss if you prefer:
    # criterion = nn.MSELoss()

    print("[DEBUG] Model architecture:")
    print(model)

    ############################################################################
    # 4.4 Training Loop
    ############################################################################
    print("[INFO] Starting training loop...\n")
    for epoch in range(NUM_EPOCHS):
        # ----------------- Training Phase -----------------
        model.train()
        total_loss = 0.0

        for batch_i, (batch_X, batch_y) in enumerate(train_loader, start=1):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()  # shape [batch_size]
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ----------------- Testing Phase -----------------
        model.eval()
        total_test_loss = 0.0
        all_abs_errors = []

        with torch.no_grad():
            for batch_i, (batch_X, batch_y) in enumerate(test_loader, start=1):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                total_test_loss += loss.item()

                # Collect absolute errors for median
                abs_errors = (predictions - batch_y).abs().cpu().numpy()
                all_abs_errors.extend(abs_errors)

        avg_test_loss = total_test_loss / len(test_loader)
        median_test_error = float(np.median(all_abs_errors))

        print(f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train MAE: {avg_train_loss:.4f}, "
              f"Test MAE: {avg_test_loss:.4f}, "
              f"Test Median Error: {median_test_error:.4f}")

    print("\n[INFO] Training completed!")
    print("=== END SCRIPT ===")

    ########################################################################
    # 4.5  Produce a 1D Histogram of Prediction Errors on Test Data
    ########################################################################
    # We gather final predictions vs. actual for test samples, then plot
    # Save the trained model
    model_save_path = "age_predictor_mlp.pth"
    torch.save(model, model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
    model.eval()
    final_preds = []
    final_ages = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).squeeze().cpu().numpy()
            final_preds.extend(preds)
            final_ages.extend(batch_y.cpu().numpy())

    final_preds = np.array(final_preds)
    final_ages = np.array(final_ages)

    # Compute the error: (predicted_age - actual_age)
    errors = final_preds - final_ages

    plt.figure(figsize=(7, 5))
    # Define bins from -50 to 50 in steps of 5 (adjust to your dataset's range)
    bins = np.arange(-25, 25, 2.5)

    plt.hist(errors, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')  # vertical line at error=0

    plt.title("Histogram of Age Prediction Errors (Test Set)")
    plt.xlabel("Error (Predicted Age - Actual Age)")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    # Save or display figure
    plt.savefig("age_error_histogram.png")
    plt.show()


###############################################################################
# 5. Entry Point
###############################################################################
if __name__ == "__main__":
    main()
