from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)
im_color = io.imread(in_dir + 'ardeche.jpg')

print(im_org.shape)
print(im_org.dtype)


# Exercise 4
# The highest value is in the white parts of the fingers as about 190 and the darkest at about 58 in the left buttom corner
def showGreyMap():
    io.imshow(im_org)
    plt.title('Metacarpal image')
    io.show()

# Exercise 5
def showColoredMap():
    io.imshow(im_org, cmap="copper")
    plt.title('Metacarpal image (with colormap)')
    io.show()

# Exercise 6
def showimShow():
    io.imshow(im_org, vmin=20, vmax=170)
    plt.title('Metacarpal image (with gray level scaling)')
    io.show()

# Exercise 7
# As there is many pixels above 170, the image is very bright
# Setting the brightness to the acutal max and min, makes it easier to see the image
def scaleImageAutomactic():
    vmin, vmax = np.min(im_org), np.max(im_org)
    io.imshow(im_org, vmin=vmin, vmax=vmax)
    plt.title('Metacarpal image (with gray level scaling)')
    io.show()

#Exercise 8
def histogram():
    histogramAddData()
    plt.hist(im_org.ravel(), bins=256)
    plt.title('Image histogram')
    io.show()

#Exercise 8.1
def histogramAddData():
    h = plt.hist(im_org.ravel(), bins=256)
    bin_no = 100
    count = h[0][bin_no]
    print(f"There are {count} pixel values in bin {bin_no}")
    
    bin_left = h[1][bin_no]
    bin_right = h[1][bin_no + 1]
    print(f"Bin edges: {bin_left} to {bin_right}")

#Exercise 9
# Finds the most common pixel value range
def histogramCommonRange():
    y, x, _ = plt.hist(im_org.ravel(), bins=256)
    max_bin_index = np.argmax(y)
    bin_left = x[max_bin_index]
    bin_right = x[max_bin_index + 1]

    print(f"Most common pixel value range: {bin_left} to {bin_right} and count: {y[max_bin_index]}")

#Exercise 10
#Pixel value at (r,c) = (110, 90)
def pixelValue():
    r = 100
    c = 50
    im_val = im_org[r, c]
    print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

    print(f"Pixel value at (110, 90): {im_org[110, 90]}")

#Exercise 11
# removes the top 30 pixels from the image
def removeTop():
    im_org[:30] = 0
    io.imshow(im_org)
    io.show()

# Makes a image where is can either be black or white
def binaryImageBlackWHite():
    mask = im_org > 150
    io.imshow(mask)
    io.show()


#Exercise 12
#Black is 1, white is 0

#Exercise 13
#Sets all pixel values above 150 to 255 (white)
def setThresholdForColor():
    mask = im_org > 150
    im_org[mask] = 255
    io.imshow(im_org)
    io.show()

#Exercise 14
#Print Color image
#(512, 512) uint8
def printColorImage():
    io.imshow(im_color)
    plt.title('Metacarpal image')
    io.show()

#Exercise 15
#RGB colors at (r, c) = (110, 90)
def printRGBColorAt():
    r = 110
    c = 90
    print(f"RGB colors at (r, c) = ({im_color[r, c]})")
   

#Exercise 16
def colorUpperHalf():
    rows = im_color.shape[0]
    r_2 = int(rows / 2)
    im_color[:r_2] = [0, 255,0]
    io.imshow(im_color)
    io.show()

colorUpperHalf()

