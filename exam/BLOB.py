import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import disk

# Load the DICOM image
dicom_file = "data/kidneys/1-189.dcm"
dicom_data = pydicom.dcmread(dicom_file)
image = dicom_data.pixel_array

# Apply Otsu's threshold
threshold = threshold_otsu(image)
binary_image = image > threshold

# Label connected components
labeled_image = measure.label(binary_image)
properties = measure.regionprops(labeled_image, cache=True)

# Filter BLOBs based on perimeter and area
min_perimeter = 400
max_perimeter = 600
max_area = 5000

areas = []
for prop in properties:
        areas.append(prop.area)

# Plot histogram of areas
plt.hist(areas, bins=30, edgecolor='black')
plt.title("Histogram of BLOB Areas")
plt.xlabel("Area")
plt.ylabel("Frequency")
plt.show()