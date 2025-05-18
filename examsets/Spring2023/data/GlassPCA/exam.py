import numpy as np
import matplotlib.pyplot as plt

# Load data
# % Glass data file should have no headers and whitespace-separated values
data = np.loadtxt('glass_data.txt', comments='%')

# % Step 1: Subtract the mean from the data (centering)
mean_vals = np.mean(data, axis=0)
data_centered = data - mean_vals

# % Step 2 & 3: Min-max scaling
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
range_vals = max_vals - min_vals
data_scaled = data_centered / range_vals

# % Compute covariance matrix
cov_matrix = np.cov(data_scaled, rowvar=False)

# % Print the value of the covariance matrix at position (0, 0)
print(f"Value of the covariance matrix at position (0, 0): {cov_matrix[0, 0]:.4f}")

# % Perform eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# % Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# % Explained variance by the first three components
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
explained_first_3 = np.sum(explained_variance_ratio[:3])
print(f"Total variance explained by the first 3 components: {explained_first_3:.4f}")

# % First value of Sodium (Na) after centering and scaling
first_na_value = data_scaled[0, 1]  # assuming Na is column 1
print(f"First value of Sodium (Na) after centering and scaling: {first_na_value:.4f}")

# % Project data onto principal components
projected_data = data_scaled @ sorted_eigenvectors

# % Compute max absolute value in the projected data
max_abs_projection = np.max(np.abs(projected_data))
print(f"Maximum absolute value in projected data: {max_abs_projection:.4f}")
