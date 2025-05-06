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
im_personal = io.imread(in_dir + 'P1010266.JPG')
im_bright = io.imread(in_dir + 'bright.JPG')
im_dark = io.imread(in_dir + 'dark.JPG')
im_darkBright = io.imread(in_dir + 'darkBright.jpg')
im_dtuSign = io.imread(in_dir + 'DTUSign1.jpg')

#print(im_org.shape)
#print(im_org.dtype)

def printShapeAndType(image):
    print(image.shape)
    print(image.dtype)


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


#Exercise 17
#In the shape (3000, 4000, 3), the 3 represents the number of color channels in the image. This typically indicates that the image is a color image with three channels: Red, Green, and Blue (RGB).
def printPersonalImage():
    printShapeAndType(im_personal)
    io.imshow(im_personal)
    plt.title('Personal image')
    io.show()
#printPersonalImage()

#Exercise 18
#Image is float34 and has gone from 4000x3000 to 750x666
# The pixel are now in range 0-1
#The pixels are in the range 0-1 instead of 0-255 because the image has been normalized. This typically happens when the image is converted to a floating-point representation where pixel values are scaled to the range [0, 1] for easier manipulation and processing.
def scalePersonalImage():
    image_resized = resize(im_personal, (im_personal.shape[0] // 4,
                       im_personal.shape[1] // 6),
                       anti_aliasing=True)
    printShapeAndType(image_resized)
    io.imshow(image_resized)
    plt.title('Personal image resized')
    io.show()
    
#scalePersonalImage()

#Exercise 19
def autoScaleTo400():
    new_height = 400
    new_width = int(im_personal.shape[1] / (im_personal.shape[0] / new_height))
    image_resized = resize(im_personal, (new_height, new_width), anti_aliasing=True)
    printShapeAndType(image_resized)
    io.imshow(image_resized)
    plt.title('Personal image resized')
    io.show() 
#autoScaleTo400()

#Exercise 19.1
def histogramOfPersonalImage():
    plt.hist(im_personal.ravel(), bins=256)
    plt.title('Personal image histogram')
    io.show()
#histogramOfPersonalImage()

#Exercise 19.2
def histogramOfPersonalImageGrey():
    im_gray = color.rgb2gray(im_personal)
    im_byte = img_as_ubyte(im_gray)
    plt.hist(im_byte.ravel(), bins=256)
    plt.title('Personal image histogram')
    io.show()
#histogramOfPersonalImageGrey()

#Exercise 20
#Image.png shows the bright image having a higher pixel value range compared to the dark image. The bright image has a pixel value range of 75-190, while the dark image has a a large amount of pixels range of 0-50.
def histBrightDarkCompare():
    plt.subplot(1, 2, 1)
    plt.hist(im_bright.ravel(), bins=256)
    plt.title('Bright image histogram')

    # Second subplot for the dark image
    plt.subplot(1, 2, 2)
    plt.hist(im_dark.ravel(), bins=256)
    plt.title('Dark image histogram')
    io.show()
#histBrightDarkCompare()

#Exercise 21
# Kinda, there is alot of dark pixels, so a crazy amount of pixels are in the range 0-20, and then barely any in the range 50 and above
def histDarkBright():
    plt.hist(im_darkBright.ravel(), bins=256)
    plt.title('Personal image histogram')
    io.show()
#histDarkBright()

#Exercise 22
def showDTUSign():
    printShapeAndType(im_dtuSign)
    io.imshow(im_dtuSign)
    plt.title('DTU sign')
    io.show()
#showDTUSign()

#Exercise 23
def showDTUSignRGB():
    r_comp = im_dtuSign[:, :, 0]
    io.imshow(r_comp)
    plt.title('DTU sign image (Red)')
    io.show()
#showDTUSignRGB()

#Exercise 23
# The DTU image is bright red, because the red channel is the only one that is shown
# The walls are bright, because they are reflecting all spectrums of light, as it is grey.
def showDTUSignRGBAll():
 # Extract the R, G, and B components
    im_dtuSign_R = im_dtuSign[:, :, 0]
    im_dtuSign_G = im_dtuSign[:, :, 1]
    im_dtuSign_B = im_dtuSign[:, :, 2]

    plt.figure(figsize=(18, 6))

    # Display the R component
    plt.subplot(1, 4, 1)
    plt.imshow(im_dtuSign_R)
    plt.title('DTU Sign - Red Component')

    # Display the G component
    plt.subplot(1, 4, 2)
    plt.imshow(im_dtuSign_G)
    plt.title('DTU Sign - Green Component')

    # Display the B component
    plt.subplot(1, 4, 3)
    plt.imshow(im_dtuSign_B)
    plt.title('DTU Sign - Blue Component')

    plt.subplot(1, 4, 4)
    plt.imshow(im_dtuSign)
    plt.title('DTU Sign')

    plt.show()
#showDTUSignRGBAll()

im_dtuSign1 = io.imread(in_dir + 'DTUSign1.jpg')

#Exercise 24
def visulaizeImageWithBlackSquare():
    im_dtuSign1[500:1000, 800:1500, :] = 0
    io.imshow(im_dtuSign1)
    plt.title('DTU sign')
    io.show()
#visulaizeImageWithBlackSquare()

#Exercise 25
def saveImageToDisk():
    io.imsave('DTUSign1_modified.png', im_dtuSign1)
#saveImageToDisk()

#Exercise 26
def blueSquareAroundDtusign():
    im_dtuSign1[1500:1700, 2300:2700, 2] = 255
    io.imsave('DTUSign1_modified.png', im_dtuSign1)
#blueSquareAroundDtusign()

#Exercise 27
def makeBonesBlue():
    im_org_rgb = color.gray2rgb(im_org)
    mask = im_org > 140
    im_org_rgb[mask] = [0, 0, 255]
    io.imshow(im_org_rgb)
    io.show()
#makeBonesBlue()

#Exercise 28
#yes the background is black, so low on the graph, the outer sides of the funger is brightest, and the middle is a little darker, so the graph is a little U shaped in the top
def sampleGreyScale():
    p = profile_line(im_org, (342, 77), (320, 160))
    plt.plot(p)
    plt.ylabel('Intensity')
    plt.xlabel('Distance along line')
    plt.show()
#sampleGreyScale()

#Exercise 29.1
def thirdDview():
    in_dir = "data/"
    im_name = "road.png"
    ds = dicom.dcmread(in_dir + im_name)
    print(ds)
    im_org = io.imread(in_dir + im_name)
    im_gray = color.rgb2gray(im_org)
    ll = 200
    im_crop = im_gray[40:40 + ll, 150:150 + ll]
    xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                        linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
#thirdDview()

#Exercise 29.2
# Rows  US: 512
# Columns   US: 512
def dicomFormat():
    in_dir = "data/"
    im_name = "1-442.dcm"
    ds = dicom.dcmread(in_dir + im_name)
    print(ds)
#dicomFormat()

#Exercise 30
def findShapeAndPixelType():
    in_dir = "data/"
    im_name = "1-442.dcm"
    ds = dicom.dcmread(in_dir + im_name)
    print(f"Rows  US: {ds.Rows}")
    print(f"Columns   US: {ds.Columns}")
    print(f"Pixel type: {ds.pixel_array.dtype}")
#findShapeAndPixelType()

#Exercise 30.1
def showSlice():
    in_dir = "data/"
    im_name = "1-442.dcm"
    ds = dicom.dcmread(in_dir + im_name)
    im = ds.pixel_array
    io.imshow(im, vmin=-1500, vmax=2000, cmap='gray')
    io.show()
showSlice()

