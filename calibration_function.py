import numpy as np
import matplotlib.pyplot as plt


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
