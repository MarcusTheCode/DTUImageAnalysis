import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color

# Step 1: Load the image
image = io.imread('rocket.png')
# Convert to grayscale if it is a color image
gray_image = color.rgb2gray(image)

# Step 2: Apply Prewitt filter (gradient magnitude)
prewitt_image = filters.prewitt(gray_image)

# Step 3: Threshold the image
threshold = 0.06
binary_image = prewitt_image > threshold

# Step 4: Count white pixels (True values in the binary image)
white_pixel_count = np.sum(binary_image)

# Show the binary image (optional)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Prewitt Filtered Image')
plt.axis('off')
plt.show()

print("Number of white pixels:", white_pixel_count)