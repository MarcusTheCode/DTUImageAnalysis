# Essential imports
import numpy as np
import math
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.morphology import disk, dilation, erosion, remove_small_objects
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import correlate
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

def image_height_in_pixels(image_height_mm, pixel_height_mm):
    """
    Computes how tall the image is on the CCD in pixels.
    """
    return image_height_mm / pixel_height_mm

def pixel_size_mm(ccd_width_mm, ccd_height_mm, pixel_width, pixel_height):
    """
    Calculates pixel size in mm.

    :return: tuple of (pixel width in mm, pixel height in mm)
    """
    return ccd_width_mm / pixel_width, ccd_height_mm / pixel_height

def apply_mean_filter(img, size):
    """
    Apply a normalized mean filter to an image.
    """
    weights = np.ones((size, size)) / (size * size)
    if img.ndim == 2:
        return correlate(img, weights)
    elif img.ndim == 3:
        return correlate(img, weights[:, :, np.newaxis])
    
def apply_median_filter(img, size):
    """
    Apply median filter with a square footprint of given size.
    """
    footprint = np.ones((size, size))
    if img.ndim == 2:
        return median(img, footprint)
    elif img.ndim == 3:
        return median(img, np.repeat(footprint[:, :, np.newaxis], img.shape[2], axis=2))
    
def apply_gaussian_filter(img, sigma):
    """
    Apply a Gaussian filter with a specified sigma value.
    """
    return gaussian(img, sigma=sigma)

def apply_prewitt_filters(img):
    """
    Apply horizontal, vertical and combined Prewitt filters.
    """
    return prewitt_h(img), prewitt_v(img), prewitt(img)

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

def compute_outline(bin_img, radius=1):
    """
    Computes outline of binary image via dilation XOR original.
    """
    dilated = dilation(bin_img, disk(radius))
    return np.logical_xor(dilated, bin_img)

def plot_comparison(original, filtered, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap='gray')
    ax2.set_title(title)
    ax2.axis('off')
    plt.show()