from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import math
from scipy.ndimage import correlate
from skimage.filters import median
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.morphology import disk, erosion, dilation, opening, closing

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()

def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline


# Read the image into im_org
im_org = io.imread('data/lego_5.png')

# Convert the image to gray scale
im_gray = color.rgb2gray(im_org)

# Find a threshold using Otsu's method
threshold = threshold_otsu(im_gray)

# Apply the threshold and generate a binary image bin_img
bin_img = im_gray > threshold

# Visualize the image using plot_comparison
plot_comparison(im_org, bin_img, 'Binary image')


# Create a disk-shaped footprint
footprint = disk(10)
# Check the size and shape of the structuring element
print(footprint)

# Apply erosion on the binary image
# Removes the white (Make black bigger)
eroded = erosion(bin_img, footprint)
plot_comparison(bin_img, eroded, 'erosion')

# Removes the black
dilated = dilation(bin_img, footprint)
plot_comparison(bin_img, dilated, 'dilation')

# Removes the white without making black bigger
opened = opening(bin_img, footprint)
plot_comparison(bin_img, opened, 'opening')

# Emphazies white, finding solid patterns of black
closed = closing(bin_img, footprint)
plot_comparison(bin_img, closed, 'closing')

# We get a black and white outline
outline = compute_outline(bin_img)
plot_comparison(bin_img, outline, 'outline')


# Perform opening with a disk of size 1
opened_small = opening(bin_img, disk(1))

# Perform closing with a disk of size 15 on the result of the opening
closed_large = closing(opened_small, disk(15))

# Compute the outline of the result
outline_final = compute_outline(closed_large)

# Visualize the final outline
plot_comparison(bin_img, outline_final, 'final outline')
# IT isolates for black removing most of the white
# Then i goes in and finds the black where it is most consentrated, leaving us with the outline

im_org = io.imread('data/lego_7.png')
im_gray = color.rgb2gray(im_org)
threshold = threshold_otsu(im_gray)
bin_img = im_gray > threshold
closed = closing(bin_img, disk(10))
outline = compute_outline(closed)
plot_comparison(bin_img, outline, 'outline')

im_org = io.imread('data/lego_3.png')
im_gray = color.rgb2gray(im_org)
threshold = threshold_otsu(im_gray)
bin_img = im_gray > threshold
closed = closing(bin_img, disk(10))
outline = compute_outline(closed)
plot_comparison(bin_img, outline, 'outline')
# We get an outline of everything


im_org = io.imread('data/lego_9.png')
im_gray = color.rgb2gray(im_org)
threshold = threshold_otsu(im_gray)
bin_img = im_gray > threshold
closed = closing(bin_img, disk(10))
outline = compute_outline(closed)
plot_comparison(im_org, bin_img, 'binary')
plot_comparison(bin_img, outline, 'outline')
# Some of the bricks stik together


im_org = io.imread('data/lego_9.png')
im_gray = color.rgb2gray(im_org)
threshold = threshold_otsu(im_gray)
bin_img = im_gray > threshold
closed = closing(bin_img, disk(5))
outline = compute_outline(closed)
eroded = erosion(closed, disk(100))
plot_comparison(im_org, bin_img, 'binary')
plot_comparison(bin_img, eroded, 'eroded')
# Some of the bricks stik together

im_org = io.imread('data/lego_9.png')
im_gray = color.rgb2gray(im_org)
threshold = threshold_otsu(im_gray)
bin_img = im_gray > threshold
closed = closing(bin_img, disk(5))
outline = compute_outline(closed)
eroded = erosion(closed, disk(100))
plot_comparison(im_org, bin_img, 'binary')
plot_comparison(bin_img, eroded, 'eroded')
# Some of the bricks stik together