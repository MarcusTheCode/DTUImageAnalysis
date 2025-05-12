from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb, rgb2gray
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 
from skimage.measure import regionprops

def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

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

def read_image(image_name):
    return io.imread("./data/" + image_name)

def apply_otsu(image):
    thres = threshold_otsu(image)
    return image > thres

# Read the image.
im_org = read_image("lego_4_small.png")
im_gray = rgb2gray(im_org)

bin_img = apply_otsu(im_gray)

""" Exercise 1 """
def exercise1():
     plot_comparison(im_org, bin_img, 'Binary Image')

""" Exercise 2 """
def exercise2():
# Remove border pixels using clear_border
    bin_img_no_border = segmentation.clear_border(bin_img)
    plot_comparison(bin_img, bin_img_no_border, 'Binary Image No Border')

footprint = disk(5)
closed = closing(bin_img, footprint)
opened = opening(closed, footprint)
""" Exercise 3 """
def exercise3():
    plot_comparison(im_org, opened, "Removed noise and close holes")

""" Exercise 4 """
def exercise4():
    label_img = measure.label(opened)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

""" Exercise 5 """
def exercise5():
    im_rgb = label2rgb(label_img)
    plot_comparison(im_org, im_rgb, "Visualization of found BLOBS")

""" Exercise 6 """
def exercise6():
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()

""" Exercise 7 """
""" Run another program """

img_org = read_image("Sample E2 - U2OS DAPI channel.tiff")
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small) 
""" Exercise 8 """
def exercise8():
    io.imshow(img_gray, vmin=20, vmax=100)
    plt.title('DAPI Stained U2OS cell nuclei')
    io.show()

    # avoid bin with value 0 due to the very large number of background pixels
    plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
    io.show()

    bin_img = apply_otsu(img_gray)

    plot_comparison(img_org, bin_img, "Binary Image")

img_c_b = segmentation.clear_border(bin_img)
label_img = measure.label(img_c_b)

""" Exercise 9 """
def exercise9():
    show_comparison(img_org, img_c_b, 'Binary Image No Border')
    image_label_overlay = label2rgb(label_img)
    show_comparison(img_org, image_label_overlay, 'Found BLOBS')


region_props = measure.regionprops(label_img)

areas = np.array([prop.area for prop in region_props])

""" Exercise 10 """
def exercise10():
    print(region_props[0].area)
    plt.hist(areas, bins=100)
    plt.show()

""" Exercise 11 """
def exercise11():
    min_area = 50
    max_area = 90

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
    
""" Exercise 12 """
def exercise12():
    perimeters = np.array([prop.perimeter for prop in region_props])
    plt.hist(areas, bins=100)
    plt.show() 
    plt.hist(perimeters, bins=100)
    plt.show()

""" Exercise 13 """
def exercise13():
    # Define your thresholds for area and circularity
    min_area = 50
    max_area = 90
    min_circularity = 0.8  # Minimum acceptable circularity
    max_circularity = 1.0  # Maximum acceptable circularity

    # Create a copy of the label_img
    label_img_filter = label_img.copy()

    # List to store circularity values
    circularities = []

    # Iterate over each region's properties
    for region in region_props:
        area = region.area
        perimeter = region.perimeter
        if perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        # Append the circularity value to the list
        circularities.append(circularity)

        # Filter based on area and circularity
        if area > max_area or area < min_area or circularity < min_circularity or circularity > max_circularity:
            # Set the pixels in the invalid areas to background (0)
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create a binary image from the filtered label image
    i_area = label_img_filter > 0

    # Plot the histogram of circularities
    plt.figure(figsize=(8, 6))
    plt.hist(circularities, bins=30, range=(0, 1))
    plt.title('Circularity Histogram')
    plt.xlabel('Circularity')
    plt.ylabel('Frequency')
    plt.show()

    # Optionally, visualize the result with filtered cells
    show_comparison(img_org, i_area, "Filtered Cells by Area and Circularity")

""" Exercise 14 """
def exercise14():
    # Define your thresholds for area and circularity
    min_area = 50
    max_area = 90
    min_circularity = 0.5  # Minimum acceptable circularity
    max_circularity = 1.0  # Maximum acceptable circularity

    # Create a copy of the label_img
    label_img_filter = label_img.copy()

    # Lists to store area and circularity values
    areas = []
    circularities = []

    # Iterate over each region's properties
    valid_nuclei_count = 0  # Initialize the count for valid nuclei
    for region in region_props:
        area = region.area
        perimeter = region.perimeter
        if perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        # Append the area and circularity values to the lists
        areas.append(area)
        circularities.append(circularity)

        # Filter based on area and circularity
        if area > max_area or area < min_area or circularity < min_circularity or circularity > max_circularity:
            # Set the pixels in the invalid areas to background (0)
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
        else:
            # Count the valid nuclei
            valid_nuclei_count += 1

    # Create a binary image from the filtered label image
    i_area = label_img_filter > 0

    # Plot Areas vs Circularity (scatter plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(areas, circularities, alpha=0.7, c='blue', edgecolors='black', s=10)
    plt.title('Area vs Circularity')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Circularity')
    plt.grid(True)
    plt.show()

    # Output the count of well-formed nuclei
    print(f"Number of well-formed nuclei: {valid_nuclei_count}")

    # Optionally, visualize the result with filtered cells
    show_comparison(img_org, i_area, "Filtered Cells by Area and Circularity")

if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    exercise5()
    exercise6()
    exercise8()
    exercise9()
    exercise10()
    exercise11()
    exercise12()
    exercise13()
