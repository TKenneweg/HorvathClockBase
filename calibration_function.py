import numpy as np
import matplotlib.pyplot as plt
from util import *
# Define the adult age threshold
adult_age = 20


# Generate a range of ages to visualize
ages = np.linspace(0, 80, 200)  # 200 points from age=0 to age=80

# Compute F(age) for each point
values = [calibFunction(age) for age in ages]

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(ages, values, label='F(age)')
plt.axvline(adult_age, color='red', linestyle='--', label='adult_age=20')

plt.xlabel('Age')
plt.ylabel('F(age)')
plt.title('Visualization of the Transformation Function')
plt.grid(True)
plt.legend()
plt.show()
