import numpy as np
from skimage import io, color

# Step 0: Load the images
background_path = 'background.png'
new_frame_path = 'new_frame.png'

background_rgb = io.imread(background_path)
new_frame_rgb = io.imread(new_frame_path)

# Step 1: Convert to grayscale
background_gray = color.rgb2gray(background_rgb)
new_frame_gray = color.rgb2gray(new_frame_rgb)

# Step 2: Update the background
alpha = 0.90
new_background = alpha * background_gray + (1 - alpha) * new_frame_gray

# Step 3: Compute the absolute difference image
diff_image = np.abs(new_frame_gray - new_background)

# Step 4: Count changed pixels (value > 0.1)
changed_pixels = np.sum(diff_image > 0.1)
print(f'Number of changed pixels: {changed_pixels}')

# Step 5: Compute the average value in the region [150:200, 150:200]
region = new_background[150:200, 150:200]
average_value = np.mean(region)
print(f'Average value in region [150:200, 150:200]: {average_value:.4f}')
