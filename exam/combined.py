
# Essential imports
"""
quick_show_orthogonal(sitk_img, origin=None, title=None)
---------------------------------------------------------
Displays orthogonal slices (axial, coronal, sagittal) of a 3D SimpleITK image.
Parameters:
- sitk_img (SimpleITK.Image): The 3D image to visualize.
- origin (tuple or None): The origin point for slicing. Defaults to the center of the image.
- title (str or None): Title for the figure. Defaults to None.
Returns:
- None
quick_overlay_slices(sitk_img_red, sitk_img_green, origin=None, title=None)
----------------------------------------------------------------------------
Displays orthogonal slices with red and green overlays for two 3D SimpleITK images.
Parameters:
- sitk_img_red (SimpleITK.Image): The red channel image.
- sitk_img_green (SimpleITK.Image): The green channel image.
- origin (tuple or None): The origin point for slicing. Defaults to the center of the image.
- title (str or None): Title for the figure. Defaults to None.
Returns:
- None
load_dicom_and_roi(dicom_path, roi_path)
----------------------------------------
Loads a DICOM image and a corresponding ROI PNG file.
Parameters:
- dicom_path (str): Path to the DICOM file.
- roi_path (str): Path to the ROI PNG file.
Returns:
- tuple: A tuple containing the DICOM image (SimpleITK.Image) and the ROI (numpy array).
blob_filter(binary_img, min_area=1500, max_area=7000, min_perimeter=300)
-------------------------------------------------------------------------
Filters binary blobs based on area and perimeter constraints.
Parameters:
- binary_img (numpy.ndarray): Binary image containing blobs.
- min_area (int): Minimum area of blobs to keep.
- max_area (int): Maximum area of blobs to keep.
- min_perimeter (int): Minimum perimeter of blobs to keep.
Returns:
- numpy.ndarray: Binary image with filtered blobs.
dice_score(binary1, binary2)
----------------------------
Computes the DICE similarity coefficient between two binary images.
Parameters:
- binary1 (numpy.ndarray): First binary image.
- binary2 (numpy.ndarray): Second binary image.
Returns:
- float: DICE score between the two binary images.
simple_threshold_segment(image, t1, t2)
---------------------------------------
Segments an image using a simple thresholding strategy.
Parameters:
- image (SimpleITK.Image): The input image.
- t1 (float): Lower threshold value.
- t2 (float): Upper threshold value.
Returns:
- numpy.ndarray: Binary image after thresholding.
morph_pipeline(binary_img, dilate1_radius=3, erode_radius=10, dilate2_radius=10)
--------------------------------------------------------------------------------
Applies a morphological pipeline (dilation, erosion, dilation) to a binary image.
Parameters:
- binary_img (numpy.ndarray): Input binary image.
- dilate1_radius (int): Radius for the first dilation.
- erode_radius (int): Radius for erosion.
- dilate2_radius (int): Radius for the second dilation.
Returns:
- numpy.ndarray: Processed binary image.
calculate_angle_with_atan2(opposite, adjacent)
----------------------------------------------
Calculates the angle in degrees using the arctangent function.
Parameters:
- opposite (float): Length of the opposite side.
- adjacent (float): Length of the adjacent side.
Returns:
- float: Angle in degrees.
solve_thin_lens(f=None, g=None, b=None)
----------------------------------------
Solves the thin lens equation: 1/f = 1/g + 1/b. Requires two of the three variables.
Parameters:
- f (float or None): Focal length. Set to None if unknown.
- g (float or None): Object distance. Set to None if unknown.
- b (float or None): Image distance. Set to None if unknown.
Returns:
- float: The missing value (f, g, or b).
Raises:
- ValueError: If more than one variable is None or insufficient data is provided.
projected_image_height(object_height_m, object_distance_m, focal_length_mm)
----------------------------------------------------------------------------
Computes the projected image height on the CCD using the pinhole camera model.
Parameters:
- object_height_m (float): Height of the object in meters.
- object_distance_m (float): Distance from the object to the camera in meters.
- focal_length_mm (float): Focal length of the camera in millimeters.
Returns:
- float: Projected image height on the CCD in millimeters.
field_of_view(ccd_width_mm, ccd_height_mm, focal_length_mm)
-----------------------------------------------------------
Calculates the horizontal and vertical field of view (FOV) in degrees.
Parameters:
- ccd_width_mm (float): Width of the CCD in millimeters.
- ccd_height_mm (float): Height of the CCD in millimeters.
- focal_length_mm (float): Focal length of the camera in millimeters.
Returns:
- tuple: Horizontal and vertical FOV in degrees.
image_height_in_pixels(image_height_mm, pixel_height_mm)
--------------------------------------------------------
Computes the height of the image on the CCD in pixels.
Parameters:
- image_height_mm (float): Height of the image in millimeters.
- pixel_height_mm (float): Height of a single pixel in millimeters.
Returns:
- float: Image height in pixels.
pixel_size_mm(ccd_width_mm, ccd_height_mm, pixel_width, pixel_height)
---------------------------------------------------------------------
Calculates the size of a pixel in millimeters.
Parameters:
- ccd_width_mm (float): Width of the CCD in millimeters.
- ccd_height_mm (float): Height of the CCD in millimeters.
- pixel_width (int): Number of pixels along the width.
- pixel_height (int): Number of pixels along the height.
Returns:
- tuple: Pixel width and height in millimeters.
apply_mean_filter(img, size)
----------------------------
Applies a normalized mean filter to an image.
Parameters:
- img (numpy.ndarray): Input image.
- size (int): Size of the filter kernel.
Returns:
- numpy.ndarray: Filtered image.
apply_median_filter(img, size)
------------------------------
Applies a median filter with a square footprint of the given size.
Parameters:
- img (numpy.ndarray): Input image.
- size (int): Size of the filter kernel.
Returns:
- numpy.ndarray: Filtered image.
apply_gaussian_filter(img, sigma)
---------------------------------
Applies a Gaussian filter with a specified sigma value.
Parameters:
- img (numpy.ndarray): Input image.
- sigma (float): Standard deviation for the Gaussian kernel.
Returns:
- numpy.ndarray: Filtered image.
apply_prewitt_filters(img)
--------------------------
Applies horizontal, vertical, and combined Prewitt filters to an image.
Parameters:
- img (numpy.ndarray): Input image.
Returns:
- tuple: Horizontal, vertical, and combined Prewitt filtered images.
detect_edges(img, filter_type='median', param=5)
-----------------------------------------------
Detects edges in an image using Prewitt and Otsu thresholding after filtering.
Parameters:
- img (numpy.ndarray): Input image.
- filter_type (str): Type of filter to apply ('median' or 'gaussian').
- param (int or float): Kernel size (for median) or sigma (for Gaussian).
Returns:
- tuple: Binary edge image and the Otsu threshold value.
Raises:
- ValueError: If an invalid filter_type is provided.
binarize_image(image_path)
--------------------------
Reads an image, converts it to grayscale, and thresholds it into a binary image.
Parameters:
- image_path (str): Path to the input image.
Returns:
- tuple: Original image, grayscale image, and binary image.
apply_erosion(bin_img, radius)
------------------------------
Applies morphological erosion to a binary image.
Parameters:
- bin_img (numpy.ndarray): Input binary image.
- radius (int): Radius of the structuring element.
Returns:
- numpy.ndarray: Eroded binary image.
apply_dilation(bin_img, radius)
-------------------------------
Applies morphological dilation to a binary image.
Parameters:
- bin_img (numpy.ndarray): Input binary image.
- radius (int): Radius of the structuring element.
Returns:
- numpy.ndarray: Dilated binary image.
apply_opening(bin_img, radius)
------------------------------
Applies morphological opening to a binary image.
Parameters:
- bin_img (numpy.ndarray): Input binary image.
- radius (int): Radius of the structuring element.
Returns:
- numpy.ndarray: Binary image after opening.
apply_closing(bin_img, radius)
------------------------------
Applies morphological closing to a binary image.
Parameters:
- bin_img (numpy.ndarray): Input binary image.
- radius (int): Radius of the structuring element.
Returns:
- numpy.ndarray: Binary image after closing.
compute_outline(bin_img, radius=1)
----------------------------------
Computes the outline of a binary image using dilation and XOR.
Parameters:
- bin_img (numpy.ndarray): Input binary image.
- radius (int): Radius of the structuring element for dilation.
Returns:
- numpy.ndarray: Binary image representing the outline.
plot_comparison(original, filtered, title)
------------------------------------------
Plots a side-by-side comparison of the original and filtered images.
Parameters:
- original (numpy.ndarray): Original image.
- filtered (numpy.ndarray): Filtered image.
- title (str): Title for the filtered image.
Returns:
- None
show_comparison(original, modified, modified_name)
--------------------------------------------------
Displays a side-by-side comparison of the original and modified images.
Parameters:
- original (numpy.ndarray): Original image.
- modified (numpy.ndarray): Modified image.
- modified_name (str): Title for the modified image.
Returns:
- None
preprocess_to_binary(image_path, slice_area=None)
-------------------------------------------------
Preprocesses an image into a binary format by grayscaling and thresholding.
Parameters:
- image_path (str): Path to the input image.
- slice_area (tuple or None): Area to slice the image (start_row, end_row, start_col, end_col). Defaults to None.
Returns:
- tuple: Grayscale image and binary image.
clean_binary_image(binary_img, close_radius=5, open_radius=5)
------------------------------------------------------------
Cleans a binary image using morphological closing and opening.
Parameters:
- binary_img (numpy.ndarray): Input binary image.
- close_radius (int): Radius for closing.
- open_radius (int): Radius for opening.
Returns:
- numpy.ndarray: Cleaned binary image.
label_and_extract_props(binary_img)
-----------------------------------
Labels connected components in a binary image and extracts their properties.
Parameters:
- binary_img (numpy.ndarray): Input binary image.
Returns:
- tuple: Labeled image and a list of region properties.
filter_by_area(label_img, region_props, min_area, max_area)
-----------------------------------------------------------
Filters labeled regions based on area constraints.
Parameters:
- label_img (numpy.ndarray): Labeled image.
- region_props (list): List of region properties.
- min_area (int): Minimum area of regions to keep.
- max_area (int): Maximum area of regions to keep.
Returns:
- numpy.ndarray: Binary image with filtered regions.
compute_circularities(region_props)
-----------------------------------
Computes the circularity of each region in the list of region properties.
Parameters:
- region_props (list): List of region properties.
Returns:
- numpy.ndarray: Array of circularity values for each region.
filter_by_circularity(region_props, min_circ, max_circ, min_area, max_area)
---------------------------------------------------------------------------
Filters regions based on circularity and area constraints.
Parameters:
- region_props (list): List of region properties.
- min_circ (float): Minimum circularity to keep.
- max_circ (float): Maximum circularity to keep.
- min_area (int): Minimum area of regions to keep.
- max_area (int): Maximum area of regions to keep.
Returns:
- int: Count of regions that meet the criteria.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
from skimage import io, color, segmentation, measure
from skimage.util import img_as_ubyte
from skimage.morphology import disk, dilation, erosion, remove_small_objects, binary_closing, binary_opening
from skimage.transform import rotate, EuclideanTransform, SimilarityTransform, warp, swirl, matrix_transform
import cv2
from skimage.draw import rectangle_perimeter
import pydicom as dicom
from skimage.measure import label, regionprops, profile_line
from skimage.filters import threshold_otsu
from scipy.ndimage import correlate
from skimage.color import label2rgb, rgb2gray, gray2rgb
from scipy.stats import norm
from skimage.morphology import disk, erosion, dilation, opening, closing
from skimage.filters import median, gaussian, prewitt, prewitt_h, prewitt_v

# Function 1: Quick 3D orthogonal viewer
def quick_show_orthogonal(sitk_img, origin=None, title=None):
    data = sitk.GetArrayFromImage(sitk_img)
    if origin is None:
        origin = np.array(data.shape) // 2
    data = img_as_ubyte(data / np.max(data))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')
    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')
    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')
    [ax.set_axis_off() for ax in axes]
    if title:
        fig.suptitle(title)
    plt.show()

# Function 2: Quick 3D overlay viewer
def quick_overlay_slices(sitk_img_red, sitk_img_green, origin=None, title=None):
    vol_r = sitk.GetArrayFromImage(sitk_img_red)
    vol_g = sitk.GetArrayFromImage(sitk_img_green)
    vol_r[vol_r < 0] = 0
    vol_g[vol_g < 0] = 0
    R = img_as_ubyte(vol_r / np.max(vol_r))
    G = img_as_ubyte(vol_g / np.max(vol_g))
    if origin is None:
        origin = np.array(R.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.stack([R[origin[0], ::-1, ::-1], G[origin[0], ::-1, ::-1], np.zeros_like(R[origin[0]])], axis=-1))
    axes[0].set_title('Axial')
    axes[1].imshow(np.stack([R[::-1, origin[1], ::-1], G[::-1, origin[1], ::-1], np.zeros_like(R[:, origin[1]])], axis=-1))
    axes[1].set_title('Coronal')
    axes[2].imshow(np.stack([R[::-1, ::-1, origin[2]], G[::-1, ::-1, origin[2]], np.zeros_like(R[:, :, origin[2]])], axis=-1))
    axes[2].set_title('Sagittal')
    [ax.set_axis_off() for ax in axes]
    if title:
        fig.suptitle(title)
    plt.show()

# Function 3: Read a DICOM and ROI PNG
def load_dicom_and_roi(dicom_path, roi_path):
    img = sitk.ReadImage(dicom_path)
    roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
    return img, roi

# Function 4: Quick BLOB filtering
def blob_filter(binary_img, min_area=1500, max_area=7000, min_perimeter=300):
    label_img = label(binary_img)
    out_img = np.zeros_like(binary_img)
    for region in regionprops(label_img):
        if min_area <= region.area <= max_area and region.perimeter >= min_perimeter:
            out_img[label_img == region.label] = 1
    return out_img

# Function 5: Simple DICE score computation
def dice_score(binary1, binary2):
    intersection = np.logical_and(binary1, binary2)
    return 2 * intersection.sum() / (binary1.sum() + binary2.sum())

# Function 6: Simple Thresholding (e.g., t1 and t2 strategy)
def simple_threshold_segment(image, t1, t2):
    data = sitk.GetArrayFromImage(image)
    binary = np.logical_and(data > t1, data < t2)
    return binary.astype(np.uint8)

# Function 7: Morphology pipeline for post-processing liver or kidney masks
def morph_pipeline(binary_img, dilate1_radius=3, erode_radius=10, dilate2_radius=10):
    binary = dilation(binary_img, disk(dilate1_radius))
    binary = erosion(binary, disk(erode_radius))
    binary = dilation(binary, disk(dilate2_radius))
    return binary

def calculate_angle_with_atan2(opposite, adjacent):
    """
    Calculate angle in degrees using math.atan2
    :param opposite: vertical leg (a)
    :param adjacent: horizontal leg (b)
    :return: angle in degrees
    """
    return math.degrees(math.atan2(opposite, adjacent))

def solve_thin_lens(f=None, g=None, b=None):
    """
    Solve the thin lens equation: 1/f = 1/g + 1/b
    You must provide two of the three variables.

    :return: The missing value.
    """
    if f is None and g is not None and b is not None:
        return 1 / (1/g + 1/b)
    elif g is None and f is not None and b is not None:
        return 1 / (1/f - 1/b)
    elif b is None and f is not None and g is not None:
        return 1 / (1/f - 1/g)
    else:
        raise ValueError("Exactly one of f, g, or b must be None, and the others must be provided.")
    
def projected_image_height(object_height_m, object_distance_m, focal_length_mm):
    """
    Computes projected image height on the CCD using the pinhole camera model.

    :param object_height_m: Height of the object in meters (e.g. Thomas's height)
    :param object_distance_m: Distance from the object to the camera in meters
    :param focal_length_mm: Focal length of the camera in millimeters
    :return: Image height on CCD in millimeters
    """
    # Convert meters to millimeters
    object_height_mm = object_height_m * 1000
    object_distance_mm = object_distance_m * 1000
    return (focal_length_mm / object_distance_mm) * object_height_mm

def field_of_view(ccd_width_mm, ccd_height_mm, focal_length_mm):
    """
    Calculates horizontal and vertical field of view in degrees.
    """
    fov_horizontal = 2 * math.degrees(math.atan(ccd_width_mm / (2 * focal_length_mm)))
    fov_vertical = 2 * math.degrees(math.atan(ccd_height_mm / (2 * focal_length_mm)))
    return fov_horizontal, fov_vertical

# Height in Pixels
def image_height_in_pixels(image_height_mm, pixel_height_mm):
    """
    Computes how tall the image is on the CCD in pixels.
    """
    return image_height_mm / pixel_height_mm

# Field-of-View Calculations
def pixel_size_mm(ccd_width_mm, ccd_height_mm, pixel_width, pixel_height):
    """
    Calculates pixel size in mm.

    :return: tuple of (pixel width in mm, pixel height in mm)
    """
    return ccd_width_mm / pixel_width, ccd_height_mm / pixel_height

# Apply Mean Filter
def apply_mean_filter(img, size):
    """
    Apply a normalized mean filter to an image.
    """
    weights = np.ones((size, size)) / (size * size)
    if img.ndim == 2:
        return correlate(img, weights)
    elif img.ndim == 3:
        return correlate(img, weights[:, :, np.newaxis])

# Apply Median Filter
def apply_median_filter(img, size):
    """
    Apply median filter with a square footprint of given size.
    """
    footprint = np.ones((size, size))
    if img.ndim == 2:
        return median(img, footprint)
    elif img.ndim == 3:
        return median(img, np.repeat(footprint[:, :, np.newaxis], img.shape[2], axis=2))

# Apply Gaussian Filter
def apply_gaussian_filter(img, sigma):
    """
    Apply a Gaussian filter with a specified sigma value.
    """
    return gaussian(img, sigma=sigma)

# Apply Prewitt Edge Filters
def apply_prewitt_filters(img):
    """
    Apply horizontal, vertical and combined Prewitt filters.
    """
    return prewitt_h(img), prewitt_v(img), prewitt(img)

def apply_prewitt_by_type(image, mode='combined'):
    """
    Apply Prewitt filter to detect image edges.

    Parameters:
        image (ndarray): Grayscale image.
        mode (str): One of ['horizontal', 'vertical', 'combined'].

    Returns:
        ndarray: Edge map using selected Prewitt filter.
    """
    if mode == 'horizontal':
        return prewitt_h(image)
    elif mode == 'vertical':
        return prewitt_v(image)
    elif mode == 'combined':
        return prewitt(image)
    else:
        raise ValueError("Invalid mode. Choose from 'horizontal', 'vertical', or 'combined'.")

def otsu_threshold_edge_detection(image, filter_type='gaussian', filter_param=2):
    """
    Detect edges using a filter followed by Prewitt and Otsu thresholding.

    Parameters:
        image (ndarray): Grayscale input image.
        filter_type (str): 'gaussian' or 'median'.
        filter_param (int or float): Size (for median) or sigma (for gaussian).

    Returns:
        tuple: (gradient_image, binary_image) after edge detection.
    """
    if filter_type == 'gaussian':
        filtered = gaussian(image, sigma=filter_param)
    elif filter_type == 'median':
        filtered = apply_median_filter(image, size=filter_param)
    else:
        raise ValueError("filter_type must be 'gaussian' or 'median'.")

    edge_img = prewitt(filtered)
    threshold = threshold_otsu(edge_img)
    binary_img = edge_img > threshold
    return edge_img, binary_img

def load_grayscale_image(path):
    """
    Load an image and convert to grayscale.

    Parameters:
        path (str): Path to the image file.

    Returns:
        ndarray: Grayscale image.
    """
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img

# Edge Detection with Preprocessing and Otsu Thresholding
def detect_edges(img, filter_type='median', param=5):
    """
    Apply either median or Gaussian filtering followed by Prewitt and Otsu edge detection.
    filter_type: 'median' or 'gaussian'
    param: kernel size (for median) or sigma (for gaussian)
    """
    if filter_type == 'median':
        filtered = apply_median_filter(img, param)
    elif filter_type == 'gaussian':
        filtered = apply_gaussian_filter(img, param)
    else:
        raise ValueError("Invalid filter_type. Use 'median' or 'gaussian'.")

    gradient = prewitt(filtered)
    threshold = threshold_otsu(gradient)
    binary = gradient > threshold
    return binary, threshold

# Basic Morphology Operations
def binarize_image(image_path):
    """Reads, grayscales, thresholds image into binary."""
    im = io.imread(image_path)
    gray = color.rgb2gray(im)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    return im, gray, binary

def apply_erosion(bin_img, radius):
    return erosion(bin_img, disk(radius))

def apply_dilation(bin_img, radius):
    return dilation(bin_img, disk(radius))

def apply_opening(bin_img, radius):
    return opening(bin_img, disk(radius))

def apply_closing(bin_img, radius):
    return closing(bin_img, disk(radius))

# Outline Extraction
def compute_outline(bin_img, radius=1):
    """
    Computes outline of binary image via dilation XOR original.
    """
    dilated = dilation(bin_img, disk(radius))
    return np.logical_xor(dilated, bin_img)

# Visualization Helper
def plot_comparison(original, filtered, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap='gray')
    ax2.set_title(title)
    ax2.axis('off')
    plt.show()

# Visualization Helper
def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified, cmap='gray')
    ax2.set_title(modified_name)
    ax2.axis('off')
    plt.show()

# Image Preprocessing for BLOB Analysis
def preprocess_to_binary(image_path, slice_area=None):
    img = io.imread(image_path)
    if slice_area:
        img = img[slice_area[0]:slice_area[1], slice_area[2]:slice_area[3]]
    if img.ndim == 3:
        img = color.rgb2gray(img)
    threshold = threshold_otsu(img)
    binary = img > threshold
    binary = segmentation.clear_border(binary)
    return img, binary

# Morphological Cleaning
def clean_binary_image(binary_img, close_radius=5, open_radius=5):
    return opening(closing(binary_img, disk(close_radius)), disk(open_radius))

# BLOB Labeling and Feature Extraction
def label_and_extract_props(binary_img):
    label_img = measure.label(binary_img)
    region_props = measure.regionprops(label_img)
    return label_img, region_props

# Image Preprocessing for BLOB Analysis
def preprocess_to_binary(image_path, slice_area=None):
    img = io.imread(image_path)
    if slice_area:
        img = img[slice_area[0]:slice_area[1], slice_area[2]:slice_area[3]]
    if img.ndim == 3:
        img = color.rgb2gray(img)
    threshold = threshold_otsu(img)
    binary = img > threshold
    binary = segmentation.clear_border(binary)
    return img, binary

# Morphological Cleaning
def preprocess_to_binary(image_path, slice_area=None):
    img = io.imread(image_path)
    if slice_area:
        img = img[slice_area[0]:slice_area[1], slice_area[2]:slice_area[3]]
    if img.ndim == 3:
        img = color.rgb2gray(img)
    threshold = threshold_otsu(img)
    binary = img > threshold
    binary = segmentation.clear_border(binary)
    return img, binary

# BLOB Labeling and Feature Extraction
def label_and_extract_props(binary_img):
    label_img = measure.label(binary_img)
    region_props = measure.regionprops(label_img)
    return label_img, region_props

# Filtering by Area
def filter_by_area(label_img, region_props, min_area, max_area):
    label_filtered = label_img.copy()
    for region in region_props:
        if region.area < min_area or region.area > max_area:
            for coord in region.coords:
                label_filtered[coord[0], coord[1]] = 0
    return label_filtered > 0

# Filtering by Circularity
def compute_circularities(region_props):
    return np.array([
        4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter != 0 else 0
        for prop in region_props
    ])

def filter_by_circularity(region_props, min_circ, max_circ, min_area, max_area):
    count = 0
    for prop in region_props:
        area, perimeter = prop.area, prop.perimeter
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter ** 2)
        if min_area <= area <= max_area and min_circ <= circ <= max_circ:
            count += 1
    return count


# Plot histogram and fitted Gaussian for spleen
def plot_pixel_histogram_with_gaussian(values, title):
    plt.hist(values, bins=50, density=True, alpha=0.6, color='g')
    mu, std = norm.fit(values)
    x = np.linspace(min(values), max(values), 100)
    plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    plt.title(title)
    plt.xlabel("Hounsfield Unit")
    plt.ylabel("Density")
    plt.show()

# Plot multiple fitted Gaussians for given organ pixel values
def plot_organ_gaussians(organ_dict, hu_range=np.arange(-200, 1000, 1.0)):
    for organ, values in organ_dict.items():
        mu, std = np.mean(values), np.std(values)
        pdf = norm.pdf(hu_range, mu, std)
        plt.plot(hu_range, pdf, label=organ)
    plt.title("Fitted Gaussians for Organs")
    plt.xlabel("Hounsfield Unit")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Define HU ranges for anatomical classes
def define_classes():
    return {
        "background": (-1000, -200),
        "fat": (-200, 0),
        "soft_tissue": (0, 300),
        "bone": (300, 2000)
    }

# Segment anatomical classes using HU ranges
def segment_anatomical_classes(img):
    thresholds = define_classes()
    segmented = {
        "background": img <= thresholds["background"][1],
        "fat": (img > thresholds["fat"][0]) & (img <= thresholds["fat"][1]),
        "soft_tissue": (img > thresholds["soft_tissue"][0]) & (img <= thresholds["soft_tissue"][1]),
        "bone": img > thresholds["bone"][0]
    }
    return segmented

# Visualize segmented classes with color mapping
def visualize_segmented_classes(img, segmented_masks):
    label_img = np.zeros_like(img, dtype=np.uint8)
    for i, cls in enumerate(["background", "fat", "soft_tissue", "bone"], start=1):
        label_img[segmented_masks[cls]] = i
    colored_img = label2rgb(label_img, bg_label=0, colors=['black', 'yellow', 'green', 'blue'])
    show_comparison(img, colored_img, 'Classified Image')

# Find optimal HU class boundaries using max-probability lookups
def find_class_boundaries_gaussian(class_params, hu_range=np.arange(-200, 1000, 1)):
    lookup = {}
    for hu in hu_range:
        best_class = max(class_params, key=lambda cls: norm.pdf(hu, *class_params[cls]))
        lookup[hu] = best_class
    # Find transitions
    transitions = []
    prev_class = None
    for hu, cls in lookup.items():
        if cls != prev_class:
            transitions.append((prev_class, cls, hu))
            prev_class = cls
    return transitions[1:]

# Segment spleen using HU thresholding and morphological operations
def spleen_segment(img, t_1=27, t_2=73, open_radius=7, close_radius=1):
    binary = (img > t_1) & (img < t_2)
    closed = binary_closing(binary, disk(close_radius))
    opened = binary_opening(closed, disk(open_radius))
    return opened

# Filter labeled regions by area and circularity
def filter_regions(label_img, min_area, max_area, circ_bounds=(0.5, 1.2)):
    region_props = measure.regionprops(label_img)
    label_img_filtered = label_img.copy()
    for region in region_props:
        area = region.area
        perimeter = region.perimeter
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0
        if not (min_area <= area <= max_area and circ_bounds[0] <= circ <= circ_bounds[1]):
            for coord in region.coords:
                label_img_filtered[coord[0], coord[1]] = 0
    return label_img_filtered > 0

# Full spleen segmentation pipeline
def spleen_finder(img, t_1=27, t_2=73, min_area=4000, max_area=4700):
    spleen_est = spleen_segment(img, t_1, t_2)
    label_img = measure.label(spleen_est)
    spleen_bin = filter_regions(label_img, min_area, max_area)
    return spleen_bin

# Full liver segmentation pipeline
def liver_finder(img, t_1=45, t_2=75, min_area=6000, max_area=10000):
    liver_est = liver_segment(img, t_1, t_2)
    label_img = measure.label(liver_est)
    liver_bin = filter_regions(label_img, min_area, max_area)
    return liver_bin

# Full kidney segmentation pipeline
def kidney_finder(img, t_1=30, t_2=70, min_area=3000, max_area=6000):
    kidney_est = kidney_segment(img, t_1, t_2)
    label_img = measure.label(kidney_est)
    kidney_bin = filter_regions(label_img, min_area, max_area)
    return kidney_bin

# Full fat segmentation pipeline
def fat_finder(img, t_1=-200, t_2=0, min_area=500, max_area=5000):
    fat_est = fat_segment(img, t_1, t_2)
    label_img = measure.label(fat_est)
    fat_bin = filter_regions(label_img, min_area, max_area)
    return fat_bin

# Compute DICE score between ground truth and prediction
def calculate_dice_score(ground_truth, prediction):
    intersection = np.sum(ground_truth * prediction)
    total = np.sum(ground_truth) + np.sum(prediction)
    return (2 * intersection) / total if total > 0 else 0

# Segment liver using HU thresholding and morphological operations
def liver_segment(img, t_1=45, t_2=75, open_radius=5, close_radius=2):
    binary = (img > t_1) & (img < t_2)
    closed = binary_closing(binary, disk(close_radius))
    opened = binary_opening(closed, disk(open_radius))
    return opened

# Segment kidney using HU thresholding and morphological operations
def kidney_segment(img, t_1=30, t_2=70, open_radius=5, close_radius=2):
    binary = (img > t_1) & (img < t_2)
    closed = binary_closing(binary, disk(close_radius))
    opened = binary_opening(closed, disk(open_radius))
    return opened

# Segment fat using HU thresholding and morphological operations
def fat_segment(img, t_1=-200, t_2=0, open_radius=3, close_radius=1):
    binary = (img > t_1) & (img <= t_2)
    closed = binary_closing(binary, disk(close_radius))
    opened = binary_opening(closed, disk(open_radius))
    return opened

# Image Rotation
def rotate_image(im, angle_deg, center=None, mode='constant', cval=0, resize=False):
    """
    Rotate an image around a given center with mode and optional resizing.
    """
    return rotate(im, angle=angle_deg, center=center, mode=mode, cval=cval, resize=resize)

# Show Comparison
def show_comparison(original, transformed, transformed_name):
    """
    Display original and transformed images side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed, cmap='gray')
    ax2.set_title(transformed_name)
    ax2.axis('off')
    plt.show()

# Create Euclidean Transform
def create_euclidean_transform(angle_deg, translation):
    """
    Create a Euclidean transform with rotation in degrees and translation.
    """
    angle_rad = np.deg2rad(angle_deg)
    return EuclideanTransform(rotation=angle_rad, translation=translation)

# Apply Transformation to Image
def apply_transform(im, tform):
    """
    Apply a geometric transformation to an image.
    """
    return warp(im, tform)

# Apply Inverse Transformation to Image
def apply_inverse_transform(im, tform):
    """
    Apply the inverse of a geometric transformation to an image.
    """
    return warp(im, tform.inverse)

# Create Similarity Transform
def create_similarity_transform(scale, angle_deg, translation):
    """
    Create a Similarity transform.
    """
    angle_rad = np.deg2rad(angle_deg)
    return SimilarityTransform(scale=scale, rotation=angle_rad, translation=translation)

# Apply Swirl Transform
def apply_swirl_transform(im, strength=10, radius=300, center=None):
    """
    Apply a swirl transformation to an image.
    """
    return swirl(im, strength=strength, radius=radius, center=center)

# Blend Two Images
def blend_images(img1, img2):
    """
    Blend two images of the same size.
    """
    return 0.5 * img_as_float(img1) + 0.5 * img_as_float(img2)

# Visualize Landmarks on Image
def visualize_landmarks(img, landmarks):
    """
    Show landmarks on an image.
    """
    plt.imshow(img)
    plt.plot(landmarks[:, 0], landmarks[:, 1], '.r', markersize=12)
    plt.show()

# Plot Two Sets of Landmarks
def plot_landmarks_comparison(src, dst, title='Landmarks Comparison'):
    """
    Show two landmark sets on the same plot.
    """
    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '-r', label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '-g', label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(title)
    plt.show()

# Compute Landmark Alignment Error
def compute_landmark_error(src, dst):
    """
    Compute sum-of-squares alignment error between landmark sets.
    """
    e = src - dst
    return np.sum(e[:, 0]**2 + e[:, 1]**2)

# Optimal Euclidean Transform from Landmarks
def optimal_euclidean_transform(src, dst):
    """
    Estimate and return the optimal Euclidean transform between landmark sets.
    """
    tform = EuclideanTransform()
    tform.estimate(src, dst)
    transformed = matrix_transform(src, tform.params)
    return tform, transformed

# Apply and Blend Transformed Source Image
def apply_and_blend_transform(src_img, dst_img, tform):
    """
    Warp and blend the source image with destination using a transform.
    """
    warped = warp(src_img, tform.inverse)
    return blend_images(warped, dst_img)

# Real-time Swirl Video Transform
def real_time_video_swirl():
    """
    Run a swirl transformation on webcam video stream in real time.
    """
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        strength = math.sin(counter / 10) * 10
        swirled = swirl(frame, strength=strength, radius=300)
        swirled = (swirled * 255).astype(np.uint8)
        swirled = cv2.cvtColor(swirled, cv2.COLOR_RGB2BGR)
        cv2.imshow('Swirled Video', swirled)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()

# Show image with optional parameters
def show_image(img, title="Image", cmap='gray', vmin=None, vmax=None):
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show histogram
def show_histogram(image, bins=256):
    plt.hist(image.ravel(), bins=bins)
    plt.title("Histogram")
    plt.show()

# Analyze a pixel at given coordinates
def analyze_pixel(image, x, y):
    print(f"Pixel value at ({x}, {y}): {image[x, y]}")

# Apply colormap to grayscale image
def apply_colormap(image, cmap, title="Colored Image"):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Scale image display using min and max
def scale_image_display(image):
    plt.imshow(image, cmap='gray', vmin=np.min(image), vmax=np.max(image))
    plt.title("Auto-scaled grayscale")
    plt.axis('off')
    plt.show()

# Compute histogram data
def compute_histogram_data(image, bins=256):
    counts, edges = np.histogram(image.ravel(), bins=bins)
    return counts, edges

# Create binary mask where image > threshold
def mask_above_threshold(image, threshold):
    return image > threshold

# Set pixels in an image to a value where mask is True
def set_pixels_with_mask(image, mask, value):
    image[mask] = value
    return image

# Extract and show color channels
def extract_color_channels(image):
    channels = ['Red', 'Green', 'Blue']
    for i in range(3):
        plt.imshow(image[:, :, i], cmap='gray')
        plt.title(channels[i])
        plt.axis('off')
        plt.show()

# Mask region in image as black
def mark_region_black(image, region):
    x1, x2, y1, y2 = region
    image[x1:x2, y1:y2] = 0
    return image

# Draw colored rectangle around region
def draw_colored_rectangle(image, region, color):
    x1, x2, y1, y2 = region
    rr, cc = rectangle_perimeter(start=(x1, y1), end=(x2-1, y2-1), shape=image.shape)
    image[rr, cc] = color
    return image

# Convert grayscale to RGB
def convert_gray_to_rgb(gray_image):
    return gray2rgb(gray_image)

# Highlight a mask in a specific color
def highlight_mask(image_rgb, mask, color):
    image_rgb[mask] = color
    return image_rgb

# Plot profile intensity
def profile_intensity(image, start, end):
    p = profile_line(image, start, end)
    plt.plot(p)
    plt.title("Intensity profile")
    plt.show()

# Read and return DICOM image and metadata
def read_dicom_image(path):
    ds = dicom.dcmread(path)
    return ds.pixel_array, ds


##Exercise 9 functions

def show_orthogonal_views(sitk_image, origin=None, title=None):
    """
    Display axial, coronal, and sagittal orthogonal slices of a 3D volume.
    If no origin is specified, it defaults to the center of the image.
    """
    data = sitk.GetArrayFromImage(sitk_image)
    if origin is None:
        origin = np.array(data.shape) // 2

    data = img_as_ubyte(data / np.max(data))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')
    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')
    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()


def overlay_orthogonal_views(image1, image2, origin=None, title=None):
    """
    Overlay orthogonal views of two 3D images in red (image1) and green (image2).
    Assumes both images are the same shape.
    """
    vol1 = sitk.GetArrayFromImage(image1)
    vol2 = sitk.GetArrayFromImage(image2)
    if vol1.shape != vol2.shape:
        raise ValueError("Image shapes must match for overlay.")
    vol1[vol1 < 0] = 0
    vol2[vol2 < 0] = 0
    if origin is None:
        origin = np.array(vol1.shape) // 2

    rgb = np.zeros(vol1.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = img_as_ubyte(vol1 / np.max(vol1))
    rgb[..., 1] = img_as_ubyte(vol2 / np.max(vol2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')
    axes[1].imshow(rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')
    axes[2].imshow(rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()


def rotation_matrix(pitch, roll=0, yaw=0):
    """
    Generate a 3D rotation matrix from pitch, roll, and yaw angles in degrees.
    """
    pitch, roll, yaw = np.deg2rad([pitch, roll, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def create_affine_matrix(pitch):
    """
    Return a 4x4 affine transformation matrix for a given pitch rotation (in degrees).
    """
    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix(pitch)
    return affine


def resample_image(volume, affine_matrix):
    """
    Apply a full 4x4 affine transformation to a SimpleITK image volume.
    """
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine_matrix[:3, :3].flatten())
    transform.SetTranslation(affine_matrix[:3, 3])  # Respect input translation
    # Do not override the center — already embedded in the matrix

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    return resampler.Execute(volume)


def save_image(image, path):
    """Save a SimpleITK image to disk."""
    sitk.WriteImage(image, path)
    print(f"Image saved to {path}")


def homogeneous_matrix_from_transform(transform):
    """
    Convert a SimpleITK transform to a 4x4 homogeneous matrix.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape(3, 3)
    matrix[:3, 3] = transform.GetTranslation()
    return matrix


def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    """
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c
    return sitk.AffineTransform(A.flatten(), t, c)


def combine_transforms_to_affine(composite_transform, center):
    """
    Convert a CompositeTransform to a single affine transformation with respect to a center.
    """
    return composite2affine(composite_transform, center)