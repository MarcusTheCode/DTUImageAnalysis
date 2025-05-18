import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Configuration ===
image_dir = ''  # <-- UPDATE this path
image_filenames = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
                   "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]

# === Step 1: Load images and preprocess ===
first_img_path = os.path.join(image_dir, image_filenames[0])
first_img = Image.open(first_img_path).convert('RGB')
img_size = first_img.size  # (width, height)

images = []
image_dict = {}
for fname in image_filenames:
    img_path = os.path.join(image_dir, fname)
    img = Image.open(img_path).convert('RGB').resize(img_size)
    img_array = np.array(img)
    images.append(img_array.flatten())
    image_dict[fname] = img_array

X = np.array(images)

# === Step 2: Compute average fish ===
avg_image = np.mean(X, axis=0)
avg_image_reshaped = avg_image.reshape((img_size[1], img_size[0], 3)).astype(np.uint8)

# Display the average image
plt.imshow(avg_image_reshaped)
plt.title("Average Fish (RGB)")
plt.axis('off')
plt.show()

# === Step 3: PCA ===
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
variance_first_two = explained_variance_ratio[0] + explained_variance_ratio[1]
print(f"Variance explained by the first two components: {variance_first_two:.4f}")

# === Step 4: SSD between neon and guppy ===
neon_img = image_dict["neon.jpg"].flatten()
guppy_img = image_dict["guppy.jpg"].flatten()
ssd = np.sum((neon_img - guppy_img) ** 2)
print(f"Pixelwise SSD between 'neon.jpg' and 'guppy.jpg': {ssd:.2f}")

# === Step 5: Find fish most different from neon in PCA space ===
neon_index = image_filenames.index("neon.jpg")
neon_pca = X_pca[neon_index]
distances = np.linalg.norm(X_pca - neon_pca, axis=1)
distances[neon_index] = -1  # Ignore self

max_index = np.argmax(distances)
most_different_fish = image_filenames[max_index]
max_distance = distances[max_index]

print(f"Fish most visually different from 'neon.jpg': {most_different_fish}")
print(f"Distance in PCA space: {max_distance:.2f}")
