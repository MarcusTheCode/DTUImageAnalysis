import numpy as np
from skimage import io, filters
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Load the image (assumes the image file is in the same directory)
# Replace 'pots.jpg' with the correct path if needed
image = io.imread('pots.jpg')

# Step 1: Extract the red channel
red_channel = image[:, :, 0]

# Step 2: Apply median filter with footprint size 10x10
filtered_red = median_filter(red_channel, size=10)

# Step 3: Threshold - foreground if value > 200
threshold_value = 200
binary_image = filtered_red > threshold_value

# Step 4: Count the number of foreground pixels
num_foreground_pixels = np.sum(binary_image)

# Display results
print("Number of foreground pixels:", num_foreground_pixels)

# Optional: visualize the binary image
plt.imshow(binary_image, cmap='gray')
plt.title("Thresholded Red Channel")
plt.axis('off')
plt.show()
