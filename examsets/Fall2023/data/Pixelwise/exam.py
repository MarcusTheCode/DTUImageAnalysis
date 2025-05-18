import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters

# Step 1: Load the image as RGB
image = io.imread('ardeche_river.jpg')  # Replace with your image path or use io.imread('https://...') for online image

# Step 2: Convert to grayscale (float image in range [0, 1])
gray_image = color.rgb2gray(image)

# Step 3: Linear grayscale histogram stretch to [0.2, 0.8]
min_val, max_val = gray_image.min(), gray_image.max()
stretched = 0.2 + ((gray_image - min_val) / (max_val - min_val)) * (0.8 - 0.2)

# Step 4: Compute average value of the histogram stretched image
average_value = np.mean(stretched)
print("Average value:", average_value)

# Step 5: Apply Prewitt horizontal filter
prewitt_h = filters.prewitt_h(stretched)

# Step 6: Compute max absolute value of Prewitt filtered image
max_abs_prewitt = np.max(np.abs(prewitt_h))
print("Max absolute value of Prewitt filtered image:", max_abs_prewitt)

# Step 7: Create binary image using threshold = average value
binary_image = stretched > average_value

# Step 8: Count foreground pixels
foreground_pixel_count = np.sum(binary_image)
print("Number of foreground pixels:", foreground_pixel_count)

# Optional: Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(stretched, cmap='gray')
axes[0].set_title('Histogram Stretched')
axes[1].imshow(np.abs(prewitt_h), cmap='gray')
axes[1].set_title('Prewitt Horizontal Filter')
axes[2].imshow(binary_image, cmap='gray')
axes[2].set_title('Binary Image')
plt.show()
