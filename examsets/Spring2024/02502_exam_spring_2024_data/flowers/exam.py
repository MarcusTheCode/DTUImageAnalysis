import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.decomposition import PCA

# Step 1: Load the first image to determine shape
folder_path = ''  # <-- Replace with actual path
first_image_path = os.path.join(folder_path, 'flower01.jpg')
first_image = imread(first_image_path)
image_shape = first_image.shape
image_size = np.prod(image_shape)

# Step 2: Load and flatten all flower images
image_filenames = [f'flower{i:02d}.jpg' for i in range(1, 16)]
images = []

for filename in image_filenames:
    image_path = os.path.join(folder_path, filename)
    image = imread(image_path)
    images.append(image.flatten())

image_matrix = np.array(images)
average_image = np.mean(image_matrix, axis=0)

# Step 3: Perform PCA
pca = PCA(n_components=5)
pca.fit(image_matrix)
projected = pca.transform(image_matrix)

# Step 4: Show average and synthetic images along first PC
def reshape_and_display(image_vector, title):
    img = image_vector.reshape(image_shape)
    img = np.clip(img / 255.0, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

reshape_and_display(average_image, "Average Image")

std_dev = np.sqrt(pca.explained_variance_[0])
pc1 = pca.components_[0]

synth_plus = average_image + 3 * std_dev * pc1
synth_minus = average_image - 3 * std_dev * pc1

reshape_and_display(synth_plus, "Synthetic Image +3PC1")
reshape_and_display(synth_minus, "Synthetic Image -3PC1")

# Step 5: Find two flowers furthest apart on PC1
pc1_coords = projected[:, 0]
min_idx = np.argmin(pc1_coords)
max_idx = np.argmax(pc1_coords)

print(f"The two flowers furthest apart along PC1 are: flower{min_idx+1:02d}.jpg and flower{max_idx+1:02d}.jpg")
print(f"Distance = {abs(pc1_coords[max_idx] - pc1_coords[min_idx]):.2f}")

# Step 6: Variance explained by first PC
explained_variance_pc1 = pca.explained_variance_ratio_[0]
print(f"Variance explained by PC1: {explained_variance_pc1:.4f} ({explained_variance_pc1 * 100:.2f}%)")

# Step 7: Match ideal flower based on PC2
ideal_image_path = os.path.join(folder_path, 'idealflower.jpg')
ideal_image = imread(ideal_image_path).flatten()

# Center the ideal image by subtracting the mean of the training set
ideal_centered = ideal_image - average_image
ideal_projected = pca.transform([ideal_image])

# Use only the second PC
pc2_coords = projected[:, 1]
ideal_pc2 = ideal_projected[0, 1]
closest_idx = np.argmin(np.abs(pc2_coords - ideal_pc2))

print(f"The flower most similar to idealflower.jpg in PC2 is: flower{closest_idx+1:02d}.jpg")
