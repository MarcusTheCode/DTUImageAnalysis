import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

# Load the DICOM image
def load_dicom_image(file_path):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    return image

# Load ROI mask and extract pixel values
def extract_values(image, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image[mask > 0]

# Dice score calculation
def dice_score(segmentation, ground_truth):
    intersection = np.logical_and(segmentation, ground_truth)
    return 2. * intersection.sum() / (segmentation.sum() + ground_truth.sum())

# Paths to the files
dicom_path = '1-162.dcm'
liver_mask_path = 'LiverROI.png'
kidney_mask_path = 'KidneyROI.png'
aorta_mask_path = 'AortaROI.png'
kidney_gt_path = 'KidneyROI.png'

# Read DICOM image
image = load_dicom_image(dicom_path)

# Extract pixel values
liver_values = extract_values(image, liver_mask_path)
kidney_values = extract_values(image, kidney_mask_path)
aorta_values = extract_values(image, aorta_mask_path)

# Compute means of regions
mean_liver = np.mean(liver_values)
mean_kidney = np.mean(kidney_values)
mean_aorta = np.mean(aorta_values)

# Compute thresholds using min-distance classification
t1 = (mean_liver + mean_kidney) / 2
t2 = (mean_kidney + mean_aorta) / 2

# Segment the image using t1 and t2
segmented = np.logical_and(image > t1, image < t2).astype(np.uint8)

# Load ground truth mask
kidney_gt = cv2.imread(kidney_gt_path, cv2.IMREAD_GRAYSCALE)
kidney_gt = (kidney_gt > 0).astype(np.uint8)

# Compute DICE score
dice = dice_score(segmented, kidney_gt)

print(f"Thresholds: t1 = {t1}, t2 = {t2}")
print(f"DICE score: {dice}")

import numpy as np
import matplotlib.pyplot as plt

# Define the points
points = np.array([
    (7, 13),
    (9, 10),
    (6, 10),
    (6, 8),
    (3, 6)
])
labels = ['(7,13)', '(9,10)', '(6,10)', '(6,8)', '(3,6)']
colors = ['blue', 'red', 'orange', 'purple', 'green']

# Theta range
theta = np.deg2rad(np.arange(0, 181))

# Plotting
plt.figure(figsize=(8, 6))
for (x, y), label, color in zip(points, labels, colors):
    rho = x * np.cos(theta) + y * np.sin(theta)
    plt.plot(np.rad2deg(theta), rho, color=color, label=label)

# Highlight intersection
plt.scatter([151], [0.29], color='black', zorder=5, label='Intersection (151°, 0.29)')

plt.title('Hough Space', fontsize=14, weight='bold')
plt.xlabel('Theta')
plt.ylabel('Rho')
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.show()
