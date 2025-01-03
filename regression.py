import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys 
from util import *

from config import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
###############################################################################
# Configuration
###############################################################################

def main():
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
    X_data = dataset.X.numpy()
    y_data = dataset.y.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    # model = Lasso(alpha=1e-2)
    # model = Ridge(alpha=5e3)
    # model = ElasticNet(alpha=0.022, l1_ratio=0.5)


    model.fit(X_train, y_train)
    # model.fit(X_train, calibFunction(y_train))


    # Print model parameters
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)
    num_large_coeffs = np.sum(np.abs(model.coef_) > 0.01)
    print(f"Number of coefficients larger than 0.01: {num_large_coeffs}")
    plt.figure()
    plt.scatter(range(len(model.coef_)), model.coef_, alpha=0.5)
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title("Scatter Plot of Model Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients_scatter_plot.png")
    plt.show()

    predstrain = model.predict(X_train)
    # predstrain = inverseCalibFunction(model.predict(X_train))
    maetrain = np.mean(np.abs(predstrain - y_train))
    mediantrain = np.median(np.abs(predstrain - y_train))
    print(f"[RESULT] Train MAE: {maetrain:.2f}")
    print(f"[RESULT] Train Median Absolute Error: {mediantrain:.2f}")
    
    preds = model.predict(X_test)
    # preds = inverseCalibFunction(model.predict(X_test))
    mae = np.mean(np.abs(preds - y_test))
    median = np.median(np.abs(preds - y_test))
    print(f"[RESULT] Test MAE: {mae:.2f}")
    print(f"[RESULT] Test Median Absolute Error: {median:.2f}")
    


    plt.figure()
    # plt.scatter(y_train, predstrain, alpha=0.5)
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.tight_layout()
    plt.savefig("age_scatter_plot.png")
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
