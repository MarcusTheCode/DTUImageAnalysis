import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import disk, closing, opening
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error

# Load DICOM and convert to Hounsfield units
def load_dicom_hu(filename):
    dicom = pydicom.dcmread(filename)
    image = dicom.pixel_array.astype(np.int16)
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

# Load ROI mask and extract pixel values
def extract_roi_pixels(image, mask_path):
    mask = imread(mask_path, as_gray=True) > 0
    return image[mask]

# Compute Dice coefficient
def dice_score(binary_image, ground_truth):
    intersection = np.logical_and(binary_image, ground_truth)
    return 2. * intersection.sum() / (binary_image.sum() + ground_truth.sum())

# Main analysis
dicom_image = load_dicom_hu('1-001.dcm')
myocardium_pixels = extract_roi_pixels(dicom_image, 'MyocardiumROI.png')
blood_pixels = extract_roi_pixels(dicom_image, 'bloodROI.png')

# Compute statistics
mu = np.mean(blood_pixels)
sigma = np.std(blood_pixels)
print(f"mu (blood) = {mu:.1f}, sigma = {sigma:.1f}")

# Class range
low = mu - 3 * sigma
high = mu + 3 * sigma
print(f"Class range: [{int(low)}, {int(high)}]")

# Segmentation
segmentation = (dicom_image > low) & (dicom_image < high)

# Morphological operations
segmentation = closing(segmentation, disk(3))
segmentation = opening(segmentation, disk(5))

# BLOB analysis
labeled = label(segmentation)
regions = regionprops(labeled)
print(f"Number of BLOBs before filtering: {len(regions)}")

filtered = np.zeros_like(segmentation, dtype=bool)
for region in regions:
    if 2000 < region.area < 5000:
        filtered[labeled == region.label] = True

# Load ground truth and compute DICE
ground_truth = imread('bloodGT.png', as_gray=True) > 0
dice = dice_score(filtered, ground_truth)
print(f"DICE score: {dice:.3f}")

# Minimum distance classifier threshold
mu_myocardium = np.mean(myocardium_pixels)
mu_blood = np.mean(blood_pixels)
threshold_min_dist = (mu_myocardium + mu_blood) / 2
print(f"Class limit (minimum-distance classifier): {threshold_min_dist:.1f}")
