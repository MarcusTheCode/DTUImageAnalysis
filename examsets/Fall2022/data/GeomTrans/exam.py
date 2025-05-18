import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import EuclideanTransform, warp, AffineTransform
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage import img_as_float

# --- PART 1: Landmark Registration ---

# 1. Define source and destination landmarks
src = np.array([[220, 55], [105, 675], [315, 675]])
dst = np.array([[100, 165], [200, 605], [379, 525]])

# 2. Compute alignment error function
def alignment_error(a, b):
    return np.sum(np.linalg.norm(a - b, axis=1)**2)

# 3. Error before registration
F_before = alignment_error(src, dst)
print("Alignment error before registration:", F_before)

# 4. Estimate Euclidean transform
tform = EuclideanTransform()
tform.estimate(src, dst)
src_transformed = tform(src)

# 5. Error after registration
F_after = alignment_error(src_transformed, dst)
print("Alignment error after registration:", F_after)

# 6. Load rocket image and warp it
rocket_img = io.imread("rocket.png")
warped = warp(rocket_img, tform.inverse)
warped_ubyte = img_as_ubyte(warped)
pixel_warped = warped_ubyte[150, 150]
print("Pixel value at (150, 150) in warped image:", pixel_warped)

# --- PART 2: Gaussian Filtering ---

# 7. Apply Gaussian filter
filtered = gaussian(rocket_img, sigma=3, channel_axis=-1)
filtered_ubyte = img_as_ubyte(filtered)
pixel_filtered = filtered_ubyte[100, 100]
print("Pixel value at (100, 100) in Gaussian filtered image:", pixel_filtered)

# --- PART 3: Rotation of CPHSun.png ---

# 8. Load CPHSun image
sun_img = io.imread("CPHSun.png")

# 9. Define rotation center and transform using degrees
rotation_center = (20, 20)
# Translate to origin → rotate → translate back
tform_rotation = AffineTransform(
    translation=(-rotation_center[0], -rotation_center[1])
) + AffineTransform(
    rotation=16  # degrees, will be handled internally if used via `rotate` (deprecated here)
) + AffineTransform(
    translation=rotation_center
)

# 10. Apply rotation
rotated = rotate(sun_img, angle=16, center=rotation_center)  # Drop preserve_range
rotated_ubyte = img_as_ubyte(rotated)  # Safe: values are in [0, 1]


# 11. Get pixel at (200, 200)
pixel_rotated = rotated_ubyte[200, 200]
print("Pixel value at (200, 200) in rotated image:", pixel_rotated)
