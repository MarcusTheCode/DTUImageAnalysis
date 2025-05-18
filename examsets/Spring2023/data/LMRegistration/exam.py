import numpy as np
from skimage import io, transform, img_as_ubyte
from skimage.color import rgba2rgb

# Step 1: Load images
# Replace with actual file paths
shoe1_path = 'shoe_1.png'
shoe2_path = 'shoe_2.png'
shoe1 = io.imread(shoe1_path)
shoe2 = io.imread(shoe2_path)

# If images are RGBA, convert to RGB
if shoe1.shape[-1] == 4:
    shoe1 = rgba2rgb(shoe1)
if shoe2.shape[-1] == 4:
    shoe2 = rgba2rgb(shoe2)

# Landmarks
src = np.array([[40, 320], [425, 120], [740, 330]])  # shoe_1
dst = np.array([[80, 320], [380, 155], [670, 300]])  # shoe_2

# Step 2: Similarity transform
tform = transform.estimate_transform('similarity', src, dst)

# Extract the scale
scale = np.sqrt(tform.params[0, 0]**2 + tform.params[0, 1]**2)
print("Scale of transform:", scale)

# Step 3: Alignment error (sum of squared distances)
def alignment_error(points1, points2):
    return np.sum((points1 - points2)**2)

F_before = alignment_error(src, dst)
src_transformed = tform(src)
F_after = alignment_error(src_transformed, dst)
print("Alignment error before:", F_before)
print("Alignment error after:", F_after)

# Step 4: Transform the source image
shoe1_registered = transform.warp(shoe1, tform.inverse, output_shape=shoe2.shape)

# Convert both images to byte format
shoe1_byte = img_as_ubyte(shoe1_registered)
shoe2_byte = img_as_ubyte(shoe2)

# Extract blue component at (200, 200)
blue_shoe1 = shoe1_byte[200, 200, 2]
blue_shoe2 = shoe2_byte[200, 200, 2]

# Calculate absolute difference
blue_diff = abs(int(blue_shoe1) - int(blue_shoe2))
print("Absolute difference in blue component:", blue_diff)
