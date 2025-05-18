import cv2
import numpy as np
from skimage.morphology import disk, dilation
import matplotlib.pyplot as plt

# Step 1: Load the image and convert from RGB to HSV
# Replace 'path_to_image.jpg' with the path to your input image
image_path = 'nike.png'
image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Step 2: Extract only the H (Hue) channel
h_channel = image_hsv[:, :, 0] / 179.0  # Normalize H to [0, 1]

# Step 3: Create a binary mask for 0.3 < H < 0.7
binary_mask = np.logical_and(h_channel > 0.3, h_channel < 0.7).astype(np.uint8)

# Step 4: Morphological dilation using disk-shaped structuring element with radius=8
selem = disk(8)
dilated_image = dilation(binary_mask, selem)

# Step 5: Count foreground pixels (value = 1)
foreground_pixel_count = np.sum(dilated_image)
print(f"Number of foreground pixels in the resulting image: {foreground_pixel_count}")

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Binary Mask (0.3 < H < 0.7)')
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('After Dilation')
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')
plt.show()
