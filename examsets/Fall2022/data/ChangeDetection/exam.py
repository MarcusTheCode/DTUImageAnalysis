import numpy as np
from skimage import io, color

# Step 1: Load images and convert to grayscale
img1 = io.imread('change1.png')  # Replace with actual path if needed
img2 = io.imread('change2.png')

gray1 = color.rgb2gray(img1)
gray2 = color.rgb2gray(img2)

# Step 2: Compute the absolute difference image
diff = np.abs(gray1 - gray2)

# Step 3: Count pixels where difference > 0.3
threshold = 0.3
changed_pixels = np.sum(diff > threshold)

# Step 4: Calculate total pixels and percentage changed
total_pixels = gray1.size
percentage_changed = (changed_pixels / total_pixels) * 100

print(f"Percentage of changed pixels: {percentage_changed:.2f}%")
