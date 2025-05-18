import numpy as np
import cv2
from skimage import io, color, img_as_float
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Step 1: Load image and convert to grayscale float
# Replace 'pixelwise.png' with the correct path if needed
image_rgb = io.imread('pixelwise.png')
image_gray = color.rgb2gray(image_rgb)
image_float = img_as_float(image_gray)

# Step 2: Linear grayscale transformation to [0.1, 0.6]
min_val, max_val = image_float.min(), image_float.max()
image_stretched = 0.1 + (image_float - min_val) * (0.6 - 0.1) / (max_val - min_val)

# Step 3: Compute threshold using Otsu's method
threshold = threshold_otsu(image_stretched)
print(f"Otsu's threshold value: {threshold:.4f}")

# Step 4: Apply threshold to create binary image
binary_image = image_stretched > threshold

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Grayscale Image')
plt.imshow(image_float, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Stretched Image')
plt.imshow(image_stretched, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
