import numpy as np
import matplotlib.pyplot as plt

# Load the data (space-delimited, ignoring lines starting with '%')
data = np.loadtxt('pistachio_data.txt', delimiter=' ', comments='%')

# Optional: Define the feature names in order
feature_names = [
    "AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY",
    "EQDIASQ", "SOLIDITY", "CONVEX_AREA", "EXTENT", "ASPECT_RATIO",
    "ROUNDNESS", "COMPACTNESS"
]

# Step 1: Subtract the mean from the data
mean_vals = np.mean(data, axis=0)
centered_data = data - mean_vals

# Step 2: Compute the standard deviation of each measurement
std_vals = np.std(centered_data, axis=0)

# Identify the feature with the smallest standard deviation
min_std_index = np.argmin(std_vals)
min_std_value = std_vals[min_std_index]
min_std_feature = feature_names[min_std_index]
print(f"Measurement with smallest standard deviation: {min_std_feature} ({min_std_value:.3f})")

# Step 3: Standardize the data (divide by std)
standardized_data = centered_data / std_vals

# Step 4: Compute the covariance matrix
cov_matrix = np.cov(standardized_data, rowvar=False)

# Find the maximum absolute value in the covariance matrix
max_abs_cov = np.max(np.abs(cov_matrix))
print(f"Maximum absolute value in the covariance matrix: {max_abs_cov:.3f}")

# Step 5: Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project the standardized data to PCA space
pca_scores = standardized_data.dot(eigenvectors)

# Print explained variance
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nExplained variance by each principal component:")
print(explained_variance_ratio)

# Determine number of components to explain at least 97% of the variance
cumulative_variance = np.cumsum(explained_variance_ratio)
num_components_97 = np.argmax(cumulative_variance >= 0.97) + 1
print(f"\nNumber of principal components to explain at least 97% variance: {num_components_97}")

# Compute distance of first nut in PCA space
first_nut_pca = pca_scores[0]
sum_of_squares = np.sum(first_nut_pca**2)
print(f"\nSum of squared projected values for the first pistachio nut: {sum_of_squares:.3f}")

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance_ratio), marker='o')
plt.axhline(y=0.97, color='r', linestyle='--', label='97% Variance')
plt.axvline(x=num_components_97-1, color='g', linestyle='--',
            label=f'{num_components_97} Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Explained Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
