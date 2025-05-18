import numpy as np
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Load the image (replace the path below with the actual image path)
image_path = 'lights.png'  # <-- Replace with the actual file name
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# 1. Rotate the image 11 degrees with a rotation center of (40, 40)
(h, w) = image.shape[:2]
rotation_center = (40, 40)
rotation_matrix = cv2.getRotationMatrix2D(rotation_center, 11, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# 2. Convert to grayscale and to float in range [0, 1]
gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

# 3. Compute Otsu's threshold (still in [0, 1] range)
threshold_value = threshold_otsu(gray_image)
binary_image = gray_image > threshold_value

# 4. Compute percentage of foreground pixels
foreground_percentage = np.mean(binary_image) * 100

print(f"Otsu's threshold (float): {threshold_value}")
print(f"Foreground pixel percentage: {foreground_percentage:.2f}%")
