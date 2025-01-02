import numpy as np
import matplotlib.pyplot as plt

# Define the adult age threshold
adult_age = 20

# Define the transformation function F(age)
def F(age):
    if age <= adult_age:
        return np.log(age + 1) - np.log(adult_age + 1)
    else:
        return (age - adult_age) / (adult_age + 1)

# Generate a range of ages to visualize
ages = np.linspace(0, 80, 200)  # 200 points from age=0 to age=80

# Compute F(age) for each point
values = [F(age) for age in ages]

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
