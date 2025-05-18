import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data from the text file
# Assuming the file is space-delimited and has no header
data = np.loadtxt("car_data.txt", comments='%', delimiter=' ')

# Column names for context (optional)
columns = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'horsepower', 'highway-mpg']

# Step 1: Subtract the mean
means = np.mean(data, axis=0)
data_centered = data - means

# Step 2: Normalize by standard deviation
std_devs = np.std(data_centered, axis=0)
normalized_data = data_centered / std_devs

# Value at row 0, column 0 after mean subtraction and normalization
value_0_0 = normalized_data[0, 0]
print(f"Value at row 0, column 0 after mean subtraction and normalization: {value_0_0:.4f}")

# Step 3: Compute covariance matrix
cov_matrix = np.cov(normalized_data, rowvar=False)

# Step 4: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
sorted_eigenvalues = eigenvalues[sorted_indices]

# Step 6: Compute the proportion of variance explained
variance_explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_variance = np.cumsum(variance_explained)

# Step 7: Project all normalized data to PCA space
projected_data = normalized_data @ sorted_eigenvectors

# Extract and print the absolute value of the first coordinate of the first car
first_coord = projected_data[0, 0]
print(f"Absolute value of the first coordinate of the first car in PCA space: {abs(first_coord):.4f}")

# Step 8: Project first three measurements to PCA space
subset_data = normalized_data[:, :3]
cov_subset = np.cov(subset_data, rowvar=False)
eigvals_subset, eigvecs_subset = np.linalg.eig(cov_subset)
sorted_idx_subset = np.argsort(eigvals_subset)[::-1]
sorted_eigvecs_subset = eigvecs_subset[:, sorted_idx_subset]

# Project to PCA space for subset
pca_proj_subset = subset_data @ sorted_eigvecs_subset

# Normalize: subtract mean and divide by standard deviation
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
normalized_data = (data - means) / stds

# PCA: covariance, eigendecomposition
cov_matrix = np.cov(normalized_data, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_vectors = eigenvectors[:, sorted_indices]

# Project full data onto PCA space
projected_data = normalized_data @ sorted_vectors

# Take the first 3 principal components for visualization
df = pd.DataFrame(projected_data[:, :3], columns=['PC1', 'PC2', 'PC3'])

# Create the pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of First Three Principal Components", y=1.02)
plt.show()
