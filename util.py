import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os 
import pickle
from config import *


def calibFunction(age):
    """Vectorized version of the transformation function F(age). 
    Works on scalars or NumPy arrays."""
    adult_age = 20
    age = np.asarray(age, dtype=float)  # Ensure array for vectorized ops
    
    # For entries <= adult_age, use log(age + 1) - log(21).
    # For entries > adult_age, use (age - 20) / 21.
    return np.where(
        age <= adult_age,
        np.log(age + 1) - np.log(adult_age + 1),
        (age - adult_age) / (adult_age + 1)
    )


def inverseCalibFunction(y):
    """Vectorized inverse of calibFunction. Works on scalars or NumPy arrays."""
    adult_age = 20
    y = np.asarray(y, dtype=float)  # Ensure array for vectorized ops
    
    # For entries <= 0, invert log part: age = 21*exp(y) - 1.
    # For entries > 0, invert linear part: age = 21*y + 20.
    return np.where(
        y <= 0,
        21 * np.exp(y) - 1,
        21 * y + adult_age
    )