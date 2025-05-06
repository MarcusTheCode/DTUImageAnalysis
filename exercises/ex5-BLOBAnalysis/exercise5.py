from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
from skimage.morphology import disk, erosion, dilation, opening, closing
from skimage import filters


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()io.imshow(img, vmin=?, vmax=?, cmap='gray')
io.show()

def show_binary():
    im_org = io.imread('data/lego_4_small.png')
    im_gray = color.rgb2gray(im_org)
    threshold = threshold_otsu(im_gray)
    bin_img = im_gray > threshold
    show_comparison(im_org, bin_img, 'Binary image')


def remove_border_blobs():
    im_org = io.imread('data/lego_4_small.png')
    im_gray = color.rgb2gray(im_org)
    threshold = threshold_otsu(im_gray)
    bin_img = im_gray > threshold
    bin_img = segmentation.clear_border(bin_img)
    closed = closing(bin_img, disk(5))
    open = opening(closed, disk(5))
    label_img = measure.label(open)
    n_labels = label_img.max()
    img_label = label2rgb(label_img, bg_label=0)
    print(f"Number of labels: {n_labels}")
    show_comparison(im_org, img_label, 'Binary image')
#remove_border_blobs()

def measure_region_props():
    im_org = io.imread('data/lego_4_small.png')
    im_gray = color.rgb2gray(im_org)
    threshold = threshold_otsu(im_gray)
    bin_img = im_gray > threshold
    bin_img = segmentation.clear_border(bin_img)
    closed = closing(bin_img, disk(5))
    open = opening(closed, disk(5))
    label_img = measure.label(open)
    n_labels = label_img.max()
    img_label = label2rgb(label_img, bg_label=0)
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()
#measure_region_props()

def cell_count():
    # Read the image
    img_org = io.imread('data/Sample E2 - U2OS DAPI channel.tiff')

    # Slice to extract smaller image
    img_small = img_org[700:1200, 900:1400]

    # Convert to grayscale if necessary
    if len(img_small.shape) == 3:
        img_small = color.rgb2gray(img_small)

    # Apply Otsu's threshold
    threshold = threshold_otsu(img_small)
    print(f"Otsu threshold: {threshold}")

    # Create binary image
    bin_img = img_small > threshold

    # Remove objects touching the border
    bin_img = segmentation.clear_border(bin_img)

    # Show original and binary images
    #show_comparison(img_small, bin_img, 'Binary image')

    label_img = measure.label(bin_img)
    image_label_overlay = label2rgb(label_img)
    region_props = measure.regionprops(label_img)
    print(region_props[0].area)
    areas = np.array([prop.area for prop in region_props])
    #plt.hist(areas, bins=150)
    #plt.show()


    min_area = 0
    max_area = 150

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
	    # Find the areas that do not fit our criteria
	    if region.area > max_area or region.area < min_area:
		    # set the pixels in the invalid areas to background
		    for cords in region.coords:
			    label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img_small, i_area, 'Found nuclei based on area')

    perimeters = np.array([prop.perimeter for prop in region_props])
    #plt.hist(perimeters, bins=150)
    #plt.show()

    show_comparison(img_small, image_label_overlay, 'Found BLOBS')
#cell_count()

def cell_count_circularity():
    img_org = io.imread('data/Sample E2 - U2OS DAPI channel.tiff')
    img_small = img_org[200:1200, 400:1400]
    if len(img_small.shape) == 3:
        img_small = color.rgb2gray(img_small)
    threshold = threshold_otsu(img_small)
    bin_img = img_small > threshold
    bin_img = segmentation.clear_border(bin_img)
    label_img = measure.label(bin_img)
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    min_area = 0
    max_area = 150

    label_img_filter = label_img
    for region in region_props:
	    # Find the areas that do not fit our criteria
	    if region.area > max_area or region.area < min_area:
		    # set the pixels in the invalid areas to background
		    for cords in region.coords:
			    label_img_filter[cords[0], cords[1]] = 0

    circularities = np.array([4 * math.pi * prop.area / (prop.perimeter ** 2) for prop in region_props])
    plt.hist(circularities, bins=150)
    plt.show()

    min_circularity = 0.5
    max_circularity = 1.0

    # Create a copy of the label_img
    label_img_filter_circularity = label_img.copy()
    well_formed_nuclei_count = 0
    for region in region_props:
        circularity = 4 * math.pi * region.area / (region.perimeter ** 2)
        if min_area <= region.area <= max_area and min_circularity <= circularity <= max_circularity:
            well_formed_nuclei_count += 1
            
    print(f"Number of well-formed nuclei: {well_formed_nuclei_count}")
        # Plot areas versus circularity
    plt.scatter(areas, circularities)
    plt.xlabel('Area')
    plt.ylabel('Circularity')
    plt.title('Area vs Circularity')
    plt.show()
    # Create binary image from the filtered label image
    i_circularity = label_img_filter_circularity > 0
    show_comparison(img_small, i_circularity, 'Found nuclei based on area and circularity')
cell_count_circularity()