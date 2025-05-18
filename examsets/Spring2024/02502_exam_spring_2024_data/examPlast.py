import numpy as np
import matplotlib.pyplot as plt

# Given values
mean_class1 = np.array([24, 3])
mean_class2 = np.array([45, 7])
covariance_matrix = np.array([[2, 0], [0, 2]])
x = np.array([30, 10])

# LDA computation
inv_cov = np.linalg.inv(covariance_matrix)
w = inv_cov @ (mean_class2 - mean_class1)
w0 = -0.5 * (mean_class2 @ inv_cov @ mean_class2 - mean_class1 @ inv_cov @ mean_class1)

# Compute decision function
y = w @ x + w0

# Classification
classification = "Class 2" if y >= 0 else "Class 1"
print(f"y(x) = {y}, Classified as: {classification}")

# Generate synthetic training data
np.random.seed(0)
class1_samples = np.random.multivariate_normal(mean_class1, covariance_matrix, 100)
class2_samples = np.random.multivariate_normal(mean_class2, covariance_matrix, 100)

# New measurements from camera (simulate)
new_samples = np.vstack((
    np.random.multivariate_normal(mean_class1, covariance_matrix, 50),
    np.random.multivariate_normal(mean_class2, covariance_matrix, 50)
))

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], label='Class 1 - Training', alpha=0.6)
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], label='Class 2 - Training', alpha=0.6)
plt.scatter(new_samples[:, 0], new_samples[:, 1], label='New Camera Samples', c='k', marker='x')
plt.scatter(x[0], x[1], c='red', label=f'Test Sample (classified as {classification})', s=100, edgecolors='k')
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('LDA Classification and Sample Distribution')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
