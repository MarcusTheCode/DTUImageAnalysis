import numpy as np
from skimage import io, color, measure, morphology
from scipy.ndimage import median_filter
from skimage.morphology import disk, square
import matplotlib.pyplot as plt

# Load the image
image = io.imread('Letters.png')

# --- Preprocessing ---
# Convert to grayscale and apply median filter
gray_image = color.rgb2gray(image)
filtered_image = median_filter(gray_image, size=8)
print("Pixel value at (100, 100):", filtered_image[100, 100])

# --- Red Letter Detection ---
R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
binary_image = (R > 100) & (G < 100) & (B < 100)
binary_image = binary_image.astype(np.uint8)

# Erosion
selem = disk(3)
eroded_image = morphology.erosion(binary_image, selem)

# Count foreground pixels
foreground_pixel_count = np.sum(eroded_image)
print("Foreground pixels after erosion:", foreground_pixel_count)

# --- BLOB Analysis ---
# Label the image
labeled_image = measure.label(eroded_image, connectivity=2)
regions = measure.regionprops(labeled_image)

# Filter regions based on area and perimeter
valid_blobs = []
for region in regions:
    area = region.area
    perimeter = region.perimeter
    if 1000 <= area <= 4000 and perimeter >= 300:
        valid_blobs.append(region)

print(f"Number of valid BLOBs (potential letters): {len(valid_blobs)}")

# Optional: Display the result
fig, ax = plt.subplots()
ax.imshow(image)
for blob in valid_blobs:
    y, x = blob.centroid
    ax.plot(x, y, 'ro')
plt.title("Detected Letters after Filtering")
plt.show()
