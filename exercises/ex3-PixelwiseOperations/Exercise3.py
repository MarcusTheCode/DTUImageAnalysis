from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import math

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "vertebra.png"

im_org = io.imread(in_dir + im_name)

im_name = "dark_background.png"

im_dark = io.imread(in_dir + im_name)

# No the full scale of the gray-scle spectrum is not used, as 
# the minimum value is 0 and the maximum value is 255
def showMinimumAndMaximumValues():
    print(f"Minimum value: {np.min(im_org)}")
    print(f"Maximum value: {np.max(im_org)}")
showMinimumAndMaximumValues()


# Exercise 2
# Yes it is bimodal as there are two peaks in the histogram
def displayImageAndHistogram():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im_org, cmap="gray")
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].hist(im_org.ravel(), bins=256, range=[0, 256])
    ax[1].set_title("Histogram")
    plt.show()
#displayImageAndHistogram()

# Exercise 3
def showMaxAndMinOfFloatImage():
    im_float = img_as_float(im_org)
    print(f"Minimum value rgb: {np.min(im_org)}")
    print(f"Maximum value rgb: {np.max(im_org)}")
    print(f"Minimum value float: {np.min(im_float)}")
    print(f"Maximum value float: {np.max(im_float)}")
#showMaxAndMinOfFloatImage()

# Exercise 4
# The Min and Max is as expected as they remain 57 and 235
def floatToUbyte():
    im_float = img_as_float(im_org)
    im_ubyte = img_as_ubyte(im_float)
    print(f"Minimum value: {np.min(im_ubyte)}")
    print(f"Maximum value: {np.max(im_ubyte)}")
#floatToUbyte()

#
def histogram_stretch(img_in):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    img_out = (img_float - min_val) * (max_desired - min_desired) / (max_val - min_val) + min_desired

    return img_as_ubyte(img_out)

def showHistogramStretch():
    im_stretched = histogram_stretch(im_org)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im_stretched, cmap="gray")
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].hist(im_stretched.ravel(), bins=256, range=[0, 256])
    ax[1].set_title("Histogram")
    plt.show()
#showHistogramStretch()

# Exercise 6
def showOriginalAndStretched():
    im_stretched = histogram_stretch(im_org)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image and histogram
    ax[0, 0].imshow(im_org, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].hist(im_org.ravel(), bins=256, range=[0, 256])
    ax[0, 1].set_title("Original Histogram")
    
    # Stretched image and histogram
    ax[1, 0].imshow(im_stretched, cmap="gray")
    ax[1, 0].set_title("Stretched Image")
    ax[1, 0].axis("off")
    ax[1, 1].hist(im_stretched.ravel(), bins=256, range=[0, 256])
    ax[1, 1].set_title("Stretched Histogram")
    
    plt.show()
#showOriginalAndStretched()

# Exercise 7
def gamma_map(img, gamma):
    """
    Applies gamma correction to the input image.
    :param img: Input image
    :param gamma: Gamma value for correction
    :return: Gamma corrected image as unsigned byte
    """
    img_float = img_as_float(img)
    img_gamma_corrected = np.power(img_float, gamma)
    return img_as_ubyte(img_gamma_corrected)

# Exercise 8
def showOriginalAndGammaCorrected(gamma):
    im_gamma_corrected = gamma_map(im_org, gamma)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image and histogram
    ax[0, 0].imshow(im_org, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].hist(im_org.ravel(), bins=256, range=[0, 256])
    ax[0, 1].set_title("Original Histogram")
    
    # Gamma corrected image and histogram
    ax[1, 0].imshow(im_gamma_corrected, cmap="gray")
    ax[1, 0].set_title(f"Gamma Corrected Image (gamma={gamma})")
    ax[1, 0].axis("off")
    ax[1, 1].hist(im_gamma_corrected.ravel(), bins=256, range=[0, 256])
    ax[1, 1].set_title("Gamma Corrected Histogram")
    
    plt.show()

# Example usage:
# showOriginalAndGammaCorrected(0.5)
# showOriginalAndGammaCorrected(2.0)

# Exercise 9
def threshold_image(img_in, thres):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The threshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    img_ubyte = img_as_ubyte(img_in)
    img_thresholded = np.where(img_ubyte > thres, 255, 0)
    return img_thresholded.astype(np.uint8)


def showOriginalAndThresholded(thres):
    im_thresholded = threshold_image(im_org, thres)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image and histogram
    ax[0, 0].imshow(im_org, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].hist(im_org.ravel(), bins=256, range=[0, 256])
    ax[0, 1].set_title("Original Histogram")
    
    # Thresholded image and histogram
    ax[1, 0].imshow(im_thresholded, cmap="gray")
    ax[1, 0].set_title(f"Thresholded Image (threshold={thres})")
    ax[1, 0].axis("off")
    ax[1, 1].hist(im_thresholded.ravel(), bins=256, range=[0, 256])
    ax[1, 1].set_title("Thresholded Histogram")
    
    plt.show()

# Example usage:
# showOriginalAndThresholded(128)

# Exercise 10
def test_thresholds():
    thresholds = [50, 100, 150, 200]
    for thres in thresholds:
        showOriginalAndThresholded(thres)

# Example usage:
#test_thresholds()


# Exercise 11
def apply_otsu_threshold(img):
    """
    Apply Otsu's thresholding method to an image.
    :param img: Input image
    :return: Thresholded image using Otsu's method
    """
    img_ubyte = img_as_ubyte(img)
    otsu_threshold = threshold_otsu(img_ubyte)
    img_thresholded = np.where(img_ubyte > otsu_threshold, 255, 0)
    return img_thresholded.astype(np.uint8), otsu_threshold

def showOriginalAndOtsuThresholded():
    im_thresholded, otsu_threshold = apply_otsu_threshold(im_org)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image and histogram
    ax[0, 0].imshow(im_org, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].hist(im_org.ravel(), bins=256, range=[0, 256])
    ax[0, 1].set_title("Original Histogram")
    
    # Otsu thresholded image and histogram
    ax[1, 0].imshow(im_thresholded, cmap="gray")
    ax[1, 0].set_title(f"Otsu Thresholded Image (threshold={otsu_threshold})")
    ax[1, 0].axis("off")
    ax[1, 1].hist(im_thresholded.ravel(), bins=256, range=[0, 256])
    ax[1, 1].set_title("Otsu Thresholded Histogram")
    
    plt.show()

# Example usage:
#showOriginalAndOtsuThresholded()

#Exercise 12
def showOriginalAndOtsuThresholdedNewImage():
    im_dark_gray = color.rgb2gray(im_dark)
    im_thresholded, otsu_threshold = apply_otsu_threshold(im_dark_gray)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image and histogram
    ax[0, 0].imshow(im_dark_gray, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].hist(im_dark_gray.ravel(), bins=256, range=[0, 1])
    ax[0, 1].set_title("Original Histogram")
    
    # Otsu thresholded image and histogram
    ax[1, 0].imshow(im_thresholded, cmap="gray")
    ax[1, 0].set_title(f"Otsu Thresholded Image (threshold={otsu_threshold})")
    ax[1, 0].axis("off")
    ax[1, 1].hist(im_thresholded.ravel(), bins=256, range=[0, 256])
    ax[1, 1].set_title("Otsu Thresholded Histogram")
    
    plt.show()

# Example usage:
#showOriginalAndOtsuThresholdedNewImage()

# Exercise 13
def detect_dtu_signs(img):
    """
    Detects the blue DTU sign in the input image.
    :param img: Input color image
    :return: Binary image with the blue sign as foreground
    """
    r_comp = img[:, :, 0]
    g_comp = img[:, :, 1]
    b_comp = img[:, :, 2]
    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
                (b_comp > 180) & (b_comp < 200)
    return segm_blue.astype(np.uint8) * 255

def showOriginalAndDetectedSigns():
    im_name = "DTUSigns2.jpg"
    im_signs = io.imread(in_dir + im_name)
    im_detected = detect_dtu_signs(im_signs)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax[0].imshow(im_signs)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Detected signs
    ax[1].imshow(im_detected, cmap="gray")
    ax[1].set_title("Detected Signs")
    ax[1].axis("off")
    
    plt.show()

# Example usage:
#showOriginalAndDetectedSigns()


# Exercise 14
def detect_dtu_signs(img, color='red'):
    """
    Detects the DTU sign in the input image based on the specified color.
    :param img: Input color image
    :param color: Color of the sign to detect ('blue' or 'red')
    :return: Binary image with the sign as foreground
    """
    r_comp = img[:, :, 0]
    g_comp = img[:, :, 1]
    b_comp = img[:, :, 2]
    
    if color == 'blue':
        segm = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
               (b_comp > 180) & (b_comp < 200)
    elif color == 'red':
        segm = (r_comp > 150) & (g_comp < 100) & (b_comp < 100)
    else:
        raise ValueError("Color must be 'blue' or 'red'")
    
    return segm.astype(np.uint8) * 255

def showOriginalAndDetectedSigns(color):
    im_name = "DTUSigns2.jpg"
    im_signs = io.imread(in_dir + im_name)
    im_detected = detect_dtu_signs(im_signs, color)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax[0].imshow(im_signs)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Detected signs
    ax[1].imshow(im_detected, cmap="gray")
    ax[1].set_title(f"Detected {color.capitalize()} Signs")
    ax[1].axis("off")
    
    plt.show()


# Example usage:
# showOriginalAndDetectedSigns('blue')
# showOriginalAndDetectedSigns('red')

# Exercise 15
def showHSVChannels():
    im_name = "DTUSigns2.jpg"
    im_signs = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_signs)
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]
    
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
    ax0.imshow(im_signs)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')
    
    fig.tight_layout()
    plt.show()

# Example usage:
#showHSVChannels()

def detect_dtu_signs_hsv(img, sign_color):
    """
    Detects the DTU sign in the input image based on the specified color in HSV space.
    :param img: Input color image
    :param color: Color of the sign to detect ('blue' or 'red')
    :return: Binary image with the sign as foreground
    """
    hsv_img = color.rgb2hsv(img)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    val_img = hsv_img[:, :, 2]
    
    if sign_color == 'blue':
        segm = (hue_img > 0.5) & (hue_img < 0.7) & (sat_img > 0.4) & (val_img > 0.2)
    elif sign_color == 'red':
        segm = ((hue_img < 0.05) | (hue_img > 0.95)) & (sat_img > 0.4) & (val_img > 0.2)
    else:
        raise ValueError("Color must be 'blue' or 'red'")
    
    return segm.astype(np.uint8) * 255

def showOriginalAndDetectedSignsHSV(sign_color):
    im_name = "DTUSigns2.jpg"
    im_signs = io.imread(in_dir + im_name)
    im_detected = detect_dtu_signs_hsv(im_signs, sign_color)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax[0].imshow(im_signs)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Detected signs
    ax[1].imshow(im_detected, cmap="gray")
    ax[1].set_title(f"Detected {sign_color.capitalize()} Signs in HSV")
    ax[1].axis("off")
    
    plt.show()

# Example usage:

# showOriginalAndDetectedSignsHSV('blue')
# showOriginalAndDetectedSignsHSV('red')



