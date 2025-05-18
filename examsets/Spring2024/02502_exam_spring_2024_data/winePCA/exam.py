import numpy as np

# Step 1: Load data from file
data_name = "wine-data.txt"  # Replace with your actual file path
x_org = np.loadtxt(data_name, comments="%", delimiter=None)

# Step 2: Extract measurements and producer labels
x = x_org[:, :13]          # First 13 columns: measurements
producer = x_org[:, 13]    # Last column: producer ID

# Step 3: Normalize data
x_mean = np.mean(x, axis=0)
x_range = np.ptp(x, axis=0)  # max - min
x_centered = x - x_mean
x_normalized = x_centered / x_range

# Step 4: PCA using covariance and eigen decomposition
cov_matrix = np.cov(x_normalized, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Project data to PCA space
x_pca = x_normalized @ eigenvectors
pc1 = x_pca[:, 0]  # First principal component

# Step 6a: Compute mean PC1 values for producers 1 and 2
avg_pc1_prod1 = np.mean(pc1[producer == 1])
avg_pc1_prod2 = np.mean(pc1[producer == 2])
mean_diff = np.abs(avg_pc1_prod1 - avg_pc1_prod2)

# Step 6b: Compute range of PC1 values
pc1_range = np.max(pc1) - np.min(pc1)

# Step 6c: Normalized alcohol value for first wine
alcohol_idx = 0  # First feature is alcohol
normalized_alcohol_first = x_normalized[0, alcohol_idx]

# Step 6d: Average value of the covariance matrix elements
cov_matrix_avg = np.mean(cov_matrix)

# Step 6e: Percent of total variance explained by the first five principal components
total_variance = np.sum(eigenvalues)
explained_variance_first_5 = np.sum(eigenvalues[:5])
percent_explained_first_5 = (explained_variance_first_5 / total_variance) * 100

# Step 7: Print results
print(f"Mean difference on PC1 between producer 1 and 2: {mean_diff:.4f}")
print(f"Range of the first principal component (max - min): {pc1_range:.4f}")
print(f"Normalized alcohol value of the first wine: {normalized_alcohol_first:.4f}")
print(f"Average value of elements in the covariance matrix: {cov_matrix_avg:.4f}")
print(f"Percent of total variance explained by first 5 PCs: {percent_explained_first_5:.2f}%")
