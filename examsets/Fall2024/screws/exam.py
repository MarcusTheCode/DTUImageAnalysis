import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import cv2
import os
from glob import glob
from scipy.spatial.distance import pdist, squareform

# --- Parameters ---
image_dir = ''  # <-- UPDATE this
image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
flattened_images = []
resized_images = []
file_names = []

# --- Read first image to determine size ---
first_img = cv2.imread(image_paths[0])
first_img_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
image_size = (first_img_rgb.shape[1], first_img_rgb.shape[0])  # width, height

print(f"Detected image size: {image_size}")

# --- Load and flatten all images to the same shape ---
for path in image_paths:
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, image_size, interpolation=cv2.INTER_AREA)
    resized_images.append(resized)
    flattened_images.append(resized.flatten())
    file_names.append(os.path.basename(path))

data_matrix = np.array(flattened_images)  # shape: (N_images, W*H*3)

# --- PCA computation ---
mean_image = np.mean(data_matrix, axis=0)
centered_data = data_matrix - mean_image

pca = PCA(n_components=7)
projected = pca.fit_transform(centered_data)

# Calculate cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Find the number of components to reach at least 44% variance
num_components_44 = np.argmax(cumulative_variance >= 0.44) + 1
print(f"Number of PCA components needed to explain at least 44% variance: {num_components_44}")

# --- Show Average RGB Image ---
average_image = mean_image.reshape((image_size[1], image_size[0], 3)).astype(np.uint8)
plt.figure(figsize=(4, 4))
plt.title("Average Image")
plt.imshow(average_image)
plt.axis('off')
plt.show()

# --- Find min/max PC1 ---
pc1_values = projected[:, 0]
min_index = np.argmin(pc1_values)
max_index = np.argmax(pc1_values)

img_min = resized_images[min_index]
img_max = resized_images[max_index]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"Min PC1: {file_names[min_index]}")
plt.imshow(img_min)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Max PC1: {file_names[max_index]}")
plt.imshow(img_max)
plt.axis('off')

plt.suptitle("Images with Smallest and Largest PC1 Value")
plt.tight_layout()
plt.show()

# --- Highlight screws_007.jpg in PCA space ---
index_007 = file_names.index('screws_007.jpg')
index_008 = file_names.index('screws_008.jpg')

plt.figure(figsize=(8, 6))
plt.scatter(projected[:, 0], projected[:, 1], label='All Screws')
plt.scatter(projected[index_007, 0], projected[index_007, 1],
            color='green', marker='*', s=200, label='screws_007.jpg')
plt.title('PCA Projection: PC1 vs PC2 (RGB)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# --- Euclidean distance between screws_007 and screws_008 ---
vec_007 = projected[index_007]
vec_008 = projected[index_008]
distance = np.linalg.norm(vec_007 - vec_008)
print(f"Euclidean distance between screws_007.jpg and screws_008.jpg in RGB PCA space: {distance:.2f}")



# Compute pairwise distances between all PCA vectors
distance_matrix = squareform(pdist(projected, metric='euclidean'))

# Fill diagonal with np.inf to ignore zero distances (self-distance)
np.fill_diagonal(distance_matrix, np.inf)

# Find the indices of the smallest non-zero distance
i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

print(f"The most similar images in PCA space are:\n{file_names[i]}\n{file_names[j]}")