import numpy as np
import pydicom
from skimage.io import imread
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score

def load_images(dicom_path, liver_path, kidney_l_path, kidney_r_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.int16)
    liver_mask = imread(liver_path, as_gray=True) > 0
    kidney_l_mask = imread(kidney_l_path, as_gray=True) > 0
    kidney_r_mask = imread(kidney_r_path, as_gray=True) > 0
    return img, liver_mask, kidney_l_mask, kidney_r_mask

def extract_region_values(img, mask):
    return img[mask]

def dice_score(mask1, mask2):
    return 2. * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

def ct_analysis_pipeline(dicom_path, liver_path, kidney_l_path, kidney_r_path):
    img, liver_mask, kidney_l_mask, kidney_r_mask = load_images(dicom_path, liver_path, kidney_l_path, kidney_r_path)

    # Kidney HU values
    kidney_l_vals = extract_region_values(img, kidney_l_mask)
    kidney_r_vals = extract_region_values(img, kidney_r_mask)
    print("Left Kidney Mean HU:", np.mean(kidney_l_vals))
    print("Right Kidney Mean HU:", np.mean(kidney_r_vals))

    # Liver HU statistics
    liver_vals = extract_region_values(img, liver_mask)
    liver_mean = np.mean(liver_vals)
    liver_std = np.std(liver_vals)
    t1 = liver_mean - liver_std
    t2 = liver_mean + liver_std
    print("Liver Mean HU:", liver_mean)
    print("Liver STD:", liver_std)
    print("Calculated Liver Thresholds: t1 =", t1, ", t2 =", t2)

    # Liver segmentation with calculated thresholds
    binary = ((img >= t1) & (img <= t2)).astype(np.uint8)

    # Morphological filtering
    binary = dilation(binary, disk(3))
    binary = erosion(binary, disk(10))
    binary = dilation(binary, disk(10))

    # BLOB analysis and filtering
    labeled = label(binary)
    result = np.zeros_like(binary)
    for region in regionprops(labeled):
        if 1500 <= region.area <= 7000 and region.perimeter >= 300:
            result[labeled == region.label] = 1

    # DICE score
    dice = dice_score(result.astype(bool), liver_mask.astype(bool))
    print("DICE Score with Calculated Thresholds:", dice)

# Example usage:
ct_analysis_pipeline("1-166.dcm", "LiverROI.png", "KidneyRoi_l.png", "KidneyRoi_r.png")
