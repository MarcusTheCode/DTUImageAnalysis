import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from skimage.segmentation import clear_border

# Step 1: Load and convert to grayscale
image = io.imread('figures.png')  # Replace with the correct path
gray_image = color.rgb2gray(image)

# Step 2: Compute Otsu's threshold
threshold = filters.threshold_otsu(gray_image)

# Step 3: Apply threshold (note: below threshold is foreground)
binary_image = gray_image < threshold

# Step 4: Remove blobs connected to the image border
cleaned_image = clear_border(binary_image)

# Step 5: Label image and compute region properties
label_image = measure.label(cleaned_image)
regions = measure.regionprops(label_image)

# Step 6: Compute area and perimeter
for i, region in enumerate(regions):
    print(f"Blob {i+1}: Area = {region.area}, Perimeter = {region.perimeter}")

# Visualize the results
fig, ax = plt.subplots()
ax.imshow(label_image, cmap='nipy_spectral')
ax.set_title('Labeled BLOBs')
plt.show()

# Optional: count the number of figures
print("Number of mini figures detected:", len(regions))

# Continue from previous code after computing 'regions'

# Count how many BLOBs have area > 13000 pixels
large_blobs = [region for region in regions if region.area > 13000]
print("Number of BLOBs with area > 13000 pixels:", len(large_blobs))

# Continue from previous code after computing 'regions'

# Count BLOBs with area > 13000
large_blobs = [region for region in regions if region.area > 13000]
print("Number of BLOBs with area > 13000 pixels:", len(large_blobs))

# Find BLOB with the largest area
if regions:
    largest_blob = max(regions, key=lambda r: r.area)
    print("Perimeter of the BLOB with the largest area:", largest_blob.perimeter)
else:
    print("No BLOBs found in the image.")

