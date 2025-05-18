import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import morphology, measure, io
from skimage.morphology import disk
from skimage.metrics import adapted_rand_error
from skimage.transform import resize

# -------------------- Load Data --------------------
# Replace with your file paths
dicom_path = '1-353.dcm'
mask_path = 'vertebra_gt.png'  # Expert mask

# Load DICOM image
ds = pydicom.dcmread(dicom_path)
image = ds.pixel_array.astype(np.int16)
if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
    image = image * ds.RescaleSlope + ds.RescaleIntercept

# Load expert mask (assumes it is binary, 0-255 or 0-1)
expert_mask = io.imread(mask_path)
if expert_mask.ndim == 3:
    expert_mask = expert_mask[:, :, 0]
expert_mask = expert_mask > 0

# Resize expert mask to match DICOM shape if needed
if expert_mask.shape != image.shape:
    expert_mask = resize(expert_mask, image.shape, order=0, preserve_range=True).astype(bool)

# -------------------- Segmentation --------------------
binary_image = image > 200
closed_image = morphology.closing(binary_image, disk(3))
label_image = measure.label(closed_image)

# Filter BLOBs by area
segmentation_mask = np.zeros_like(label_image, dtype=bool)
blob_areas = []
for region in measure.regionprops(label_image):
    blob_areas.append(region.area)
    if region.area >= 500:
        segmentation_mask[label_image == region.label] = True

# -------------------- Statistics --------------------
# Histogram using expert mask
vertebra_pixels = image[expert_mask]
plt.hist(vertebra_pixels, bins=100)
plt.title("HU Histogram in Expert Mask")
plt.xlabel("Hounsfield Units")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Mean and Std in segmentation
segmented_pixels = image[segmentation_mask]
mean_hu = np.mean(segmented_pixels)
std_hu = np.std(segmented_pixels)
print(f"Mean HU (segmentation): {mean_hu:.2f}")
print(f"Std HU (segmentation): {std_hu:.2f}")

# Min and Max BLOB areas
min_area = np.min(blob_areas) if blob_areas else 0
max_area = np.max(blob_areas) if blob_areas else 0
print(f"Minimum BLOB area: {min_area} pixels")
print(f"Maximum BLOB area: {max_area} pixels")

# -------------------- DICE Score --------------------
# Compute DICE = 2 * |A ∩ B| / (|A| + |B|)
intersection = np.logical_and(segmentation_mask, expert_mask).sum()
dice_score = 2 * intersection / (segmentation_mask.sum() + expert_mask.sum())
print(f"DICE Score: {dice_score:.4f}")
