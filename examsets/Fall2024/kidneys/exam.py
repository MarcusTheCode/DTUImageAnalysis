import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.io import imread
from skimage.morphology import disk, closing

def compute_dice(seg1, seg2):
    intersection = np.logical_and(seg1, seg2)
    return 2. * intersection.sum() / (seg1.sum() + seg2.sum())

# Step 1: Load the DICOM slice
dicom = pydicom.dcmread("1-189.dcm")
image = dicom.pixel_array

# Step 2: Threshold to create binary image
binary = np.logical_and(image > 100, image < 250)

# Step 3: Perform BLOB analysis
label_image = measure.label(binary, connectivity=2)
regions = measure.regionprops(label_image)

# Step 4: Filter BLOBs based on area and perimeter
filtered = np.zeros_like(binary, dtype=bool)
for region in regions:
    area = region.area
    perimeter = region.perimeter
    if 2000 < area < 5000 and perimeter > 100:
        filtered[label_image == region.label] = True

# Step 5: Morphological closing
closed = closing(filtered, disk(3))

# Step 6: Load expert annotation
gt = imread("kidneys_gt.png") > 0  # Ensure binary mask

# Step 7: Compute DICE score
dice = compute_dice(closed, gt)
print(f"DICE score: {dice:.3f}")

# Step 8: Compute total physical area
pixel_area_mm2 = 0.78 * 0.78
total_area_mm2 = np.sum(closed) * pixel_area_mm2
total_area_cm2 = total_area_mm2 / 100 
print(f"Total kidney area: {total_area_cm2:.2f} cm²")

# Optional: Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("CT Slice")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Kidney Segmentation")
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.imshow(gt, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 9: Compute the median HU value under the segmentation
kidney_hu_values = image[closed]
median_hu = np.median(kidney_hu_values)
print(f"Median Hounsfield Unit (HU): {median_hu:.1f}")


# Load the DICOM slice
dicom = pydicom.dcmread("1-189.dcm")
image = dicom.pixel_array

# Threshold to create binary image
binary = np.logical_and(image > 100, image < 250)

# Perform BLOB analysis
label_image = measure.label(binary, connectivity=2)
regions = measure.regionprops(label_image)

# Filter BLOBs by perimeter and max area, then collect their areas
qualified_areas = []
print("Qualified BLOBs (perimeter in [400,600] and area < 5000):")
for region in regions:
    area = region.area
    perimeter = region.perimeter
    if 400 <= perimeter <= 600 and area < 5000:
        qualified_areas.append(area)
        print(f"Area: {area}, Perimeter: {perimeter:.2f}")

# Suggest minimum area to keep both kidneys
if len(qualified_areas) >= 2:
    sorted_areas = sorted(qualified_areas, reverse=True)
    min_area_threshold = sorted_areas[1]  # second-largest to keep top 2
    print(f"\nSuggested minimum area threshold: {min_area_threshold}")
else:
    print("\nNot enough BLOBs matched criteria to determine minimum area.")