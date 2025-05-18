import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# 1. Load the RGB images
frame_1 = cv2.imread('frame_1.jpg')
frame_2 = cv2.imread('frame_2.jpg')

# Check if images are loaded
if frame_1 is None or frame_2 is None:
    raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

# Convert from BGR (OpenCV default) to RGB
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

# 2. Convert RGB to HSV
hsv1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2HSV)
hsv2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2HSV)

# 3. Extract the S channel and scale by 255
s1 = hsv1[:, :, 1].astype(np.float32)
s2 = hsv2[:, :, 1].astype(np.float32)

# 4. Compute the absolute difference image
diff = np.abs(s1 - s2)

# 5. Compute average and standard deviation
mean_val = np.mean(diff)
std_val = np.std(diff)

# 6. Compute threshold
threshold = mean_val + 2 * std_val

print(f"Mean value: {mean_val}, Standard deviation: {std_val}, Threshold: {threshold}")

# 7. Create binary change image
binary_change = (diff > threshold).astype(np.uint8)

# 8. Count changed pixels
changed_pixels = np.sum(binary_change)

# 9. BLOB analysis
label_image = label(binary_change, connectivity=2)
regions = regionprops(label_image)

# 10. Find largest BLOB
if regions:
    largest_blob = max(regions, key=lambda r: r.area)
    print(f"Largest BLOB area: {largest_blob.area}")
else:
    print("No BLOBs detected.")

print(f"Number of changed pixels: {changed_pixels}")

# Optional: Display the binary change image
plt.imshow(binary_change, cmap='gray')
plt.title('Binary Change Image')
plt.axis('off')
plt.show()
