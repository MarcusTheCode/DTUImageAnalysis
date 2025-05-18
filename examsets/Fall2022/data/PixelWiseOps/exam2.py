import numpy as np
from skimage import io, color, filters, morphology

# Load the image (replace with the correct path)
image = io.imread('pixelwise.png')

# Step 1: Convert RGB to HSV
hsv_image = color.rgb2hsv(image)

# Step 2: Extract the S (saturation) channel
s_channel = hsv_image[:, :, 1]

# Step 3: Compute Otsu's threshold
threshold = filters.threshold_otsu(s_channel)

# Step 4: Threshold the S channel
binary_image = s_channel > threshold

# Step 5: Perform morphological erosion with a disk of radius 4
selem = morphology.disk(4)
eroded_image = morphology.erosion(binary_image, selem)

# Count the number of foreground pixels
foreground_pixel_count = np.sum(eroded_image)
print("Foreground pixels:", foreground_pixel_count)
