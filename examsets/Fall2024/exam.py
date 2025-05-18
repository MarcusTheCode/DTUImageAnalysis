import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load the dataset
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Standardize the data: subtract mean and divide by std
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_scaled = (x - x_mean) / x_std

# Compute the covariance matrix
cov_matrix = np.cov(x_scaled.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data onto the first principal component
pc1 = eigenvectors[:, 0]
x_pca = x_scaled @ pc1

# Classify based on the sign of the first principal component projection
predicted = (x_pca < 0).astype(int)

# Compute accuracy
accuracy = accuracy_score(target, predicted)
print(f"Classifier accuracy: {accuracy * 100:.2f}%")

# Load breast cancer data
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Compute and display number of observations and features
n_observations, n_features = x.shape
print(f"Number of observations: {n_observations}")
print(f"Number of features: {n_features}")

# Standardize the data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_scaled = (x - x_mean) / x_std

# Compute the covariance matrix
cov_matrix = np.cov(x_scaled.T)

# Perform PCA using np.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data to PCA space
x_pca = x_scaled @ eigenvectors

# Output first few projected points (optional)
print("First few projections on PCA components:")
print(x_pca[:5])

import numpy as np
from sklearn.datasets import load_breast_cancer

# Load breast cancer data
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Standardize the data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_scaled = (x - x_mean) / x_std

# Compute the covariance matrix
cov_matrix = np.cov(x_scaled.T)

# Perform PCA using np.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data to PCA space
x_pca = x_scaled @ eigenvectors

# Classify based on the first principal component
# Negative projection -> classified as positive (without cancer)
positive_classified = (x_pca[:, 0] < 0).astype(int)

# Count how many are classified as positive
positive_count = np.sum(positive_classified)
print(f"Number of samples classified as positive (without cancer): {positive_count}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load breast cancer data
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Standardize the data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_scaled = (x - x_mean) / x_std

# Compute the covariance matrix
cov_matrix = np.cov(x_scaled.T)

# Perform PCA using np.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data onto the first two principal components
x_pca = x_scaled @ eigenvectors
pc1 = x_pca[:, 0]
pc2 = x_pca[:, 1]

# Plot the first two principal components
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pc1, pc2, c=target, cmap='bwr', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection of Breast Cancer Data')
plt.legend(*scatter.legend_elements(), title="Diagnosis (1=No Cancer, 0=Cancer)")
plt.grid(True)
plt.show()

import numpy as np
from sklearn.datasets import load_breast_cancer

# Load breast cancer data
breast = load_breast_cancer()
x = breast.data
target = breast.target

# Standardize the data
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_scaled = (x - x_mean) / x_std

# Compute the covariance matrix
cov_matrix = np.cov(x_scaled.T)

# Perform PCA using np.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data onto the first principal component
pc1_projections = x_scaled @ eigenvectors[:, 0]

# Compute average projections for each class
positive_avg = np.mean(pc1_projections[target == 1])  # Without cancer
negative_avg = np.mean(pc1_projections[target == 0])  # With cancer

# Output the results
print(f"Average projection for positive (no cancer): {positive_avg:.2f}")
print(f"Average projection for negative (with cancer): {negative_avg:.2f}")
