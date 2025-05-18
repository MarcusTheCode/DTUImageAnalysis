import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# --- Step 1: Load and preprocess images ---
def load_images(image_folder):
    images = []
    filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for filename in filenames:
        image_path = os.path.join(image_folder, filename)
        image = imread(image_path)
        images.append(image)
    return np.array(images), filenames

# Load the pizza images
image_folder = 'training'
images, image_names = load_images(image_folder)

# Flatten images to vectors
num_images = len(images)
image_shape = images[0].shape
flattened_images = images.reshape((num_images, -1))

# --- Step 2: Compute average pizza ---
mean_image = np.mean(flattened_images, axis=0)

# --- Step 3: Perform PCA ---
pca = PCA(n_components=5)
pca.fit(flattened_images)

# --- Step 4: Measure variation explained ---
explained_variance_ratio = pca.explained_variance_ratio_
first_pc_variation = explained_variance_ratio[0]
print(f"The first principal component explains {first_pc_variation * 100:.2f}% of the total variation.")

# --- Step 5: Find pizza farthest from the average ---
differences = flattened_images - mean_image
ssd = np.sum(differences**2, axis=1)
farthest_index = np.argmax(ssd)
print(f"The pizza visually furthest from the average is: {image_names[farthest_index]}")

# --- Step 6: Find signature pizzas based on PCA projection ---
projected = pca.transform(flattened_images)
first_pc_values = projected[:, 0]
max_pc_index = np.argmax(first_pc_values)
min_pc_index = np.argmin(first_pc_values)

print(f"Signature pizza (positive PC1): {image_names[max_pc_index]}")
plt.imshow(images[max_pc_index])
plt.title("Signature Pizza (Max PC1)")
plt.axis('off')
plt.show()

print(f"Signature pizza (negative PC1): {image_names[min_pc_index]}")
plt.imshow(images[min_pc_index])
plt.title("Signature Pizza (Min PC1)")
plt.axis('off')
plt.show()

# --- Step 7: Find most similar pizza to super_pizza.png ---
super_pizza = imread('super_pizza.png')
super_pizza_flat = super_pizza.reshape(1, -1)
super_pizza_pca = pca.transform(super_pizza_flat)

# Find the closest pizza in PCA space
distances = euclidean_distances(super_pizza_pca, projected)
closest_index = np.argmin(distances)

print(f"The most similar pizza to super_pizza.png is: {image_names[closest_index]}")
plt.imshow(images[closest_index])
plt.title("Closest Pizza to Super Pizza")
plt.axis('off')
plt.show()
