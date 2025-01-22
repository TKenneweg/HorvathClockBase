import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys 
from mlp import MethylationDataset, AgePredictorMLP

from config import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
###############################################################################
# Configuration
###############################################################################

def main():
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)

    train_indices = np.load("train_indices.npy", allow_pickle=True)
    test_indices = np.load("test_indices.npy", allow_pickle=True)
    X_data = dataset.X.numpy()
    y_data = dataset.y.numpy()
    X_train = X_data[train_indices]
    y_train = y_data[train_indices]
    X_test = X_data[test_indices]
    y_test = y_data[test_indices]


    model = LinearRegression()
    # model = Lasso(alpha=1e-2) #linear regression with L1 regularization
    # model = Ridge(alpha=5e2) #linear regression with L2 regularization
    # model = ElasticNet(alpha=0.022, l1_ratio=0.5) #linear regression with L1 and L2 regularization


    model.fit(X_train, y_train)



    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    median = np.median(np.abs(preds - y_test))
    print(f"[RESULT] Test MAE: {mae:.2f}")
    print(f"[RESULT] Test Median Absolute Error: {median:.2f}")
    

    plt.figure()
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.tight_layout()
    plt.savefig("age_scatter_plot.png")
    plt.show()


    # Print model parameters
    print("\n#####################################\n")
    print("Model coefficients:", model.coef_)
    num_large_coeffs = np.sum(np.abs(model.coef_) > 0)
    print(f"Number of coefficients larger than 0: {num_large_coeffs}")
    plt.figure()
    plt.scatter(range(len(model.coef_)), model.coef_, alpha=0.5)
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title("Scatter Plot of Model Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients_scatter_plot.png")
    plt.show()


if __name__ == "__main__":
    main()


    # # 7. Plot a histogram of errors (pred - actual)
    # errors = preds - y_test
    # plt.hist(errors, bins=20, edgecolor="black")
    # plt.axvline(x=0, color="red", linestyle="--")
    # plt.title("Histogram of Age Prediction Errors (Linear Regression)")
    # plt.xlabel("Error (Predicted - Actual)")
    # plt.ylabel("Number of Samples")
    # plt.tight_layout()
    # plt.show()
