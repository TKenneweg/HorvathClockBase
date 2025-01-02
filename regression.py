import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

###############################################################################
# Configuration
###############################################################################
SERIES_NAMES = ["GSE41037", "GSE15745"]  # Add more if desired
DATA_FOLDER = "./data"
TEST_SIZE = 0.2
RANDOM_STATE = 42  # For reproducible splits

def main():
    # 1. Gather all pickle files and load dictionaries
    all_dicts = []
    for series_id in SERIES_NAMES:
        series_subfolder = os.path.join(DATA_FOLDER, series_id)
        if not os.path.isdir(series_subfolder):
            continue
        pkl_files = [f for f in os.listdir(series_subfolder) if f.endswith(".pkl")]
        for pkl_file in pkl_files:
            pkl_path = os.path.join(series_subfolder, pkl_file)
            with open(pkl_path, "rb") as f:
                sample_dict = pickle.load(f)
            # Skip if 'age' is missing or NaN
            age = sample_dict.get("age", None)
            if age is None or (isinstance(age, float) and np.isnan(age)):
                continue
            all_dicts.append(sample_dict)

    if not all_dicts:
        print("No valid data found. Exiting.")
        return

    # 2. Identify probe keys (excluding 'age')
    all_keys = list(all_dicts[0].keys())
    all_keys.remove("age")

    # 3. Build feature matrix X and target vector y
    num_samples = len(all_dicts)
    num_probes = len(all_keys)
    X_data = np.zeros((num_samples, num_probes), dtype=np.float32)
    y_data = np.zeros(num_samples, dtype=np.float32)

    for i, dct in enumerate(all_dicts):
        for j, key in enumerate(all_keys):
            val = dct[key]
            # Simple handling for missing / non-finite data
            if val is None or not np.isfinite(val):
                val = 0.5
            X_data[i, j] = val
        y_data[i] = dct["age"]

    print(f"[INFO] Loaded {num_samples} samples, each with {num_probes} probes.")

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5. Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Print model parameters
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)
    # 6. Evaluate
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"[RESULT] Test MAE: {mae:.2f}")

    # 7. Plot a histogram of errors (pred - actual)
    errors = preds - y_test
    plt.hist(errors, bins=20, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.title("Histogram of Age Prediction Errors (Linear Regression)")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
