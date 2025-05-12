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

input_img = np.arange(25).reshape(5, 5)
print(input_img)

weights = [[0, 1, 0],
		   [1, 2, 1],
		   [0, 1, 0]]

res_img = correlate(input_img, weights)

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "Gaussian.png"

im_gaus = io.imread(in_dir + im_name)

im_salt = io.imread(in_dir + "SaltPepper.png")

#Exercise 1
#print(res_img[3][3])

#Exercise 2
# The differences in the image is that the edges are different
# Reflect mode: The image is reflected at the edges
# Constant mode: The image is padded with a constant value
def reflectAndConstant():
    print("Reflect: \n", correlate(input_img, weights, mode='reflect'))
    print("Constant: \n", correlate(input_img, weights, mode='constant', cval=10))
#reflectAndConstant()

#Exercise 3
# The image is blurred
# The dark pixels gets more watered out
def showImageWithAndWithoutFilter():
    # Update to make it more blurry
    size = 10
    # Two dimensional filter filled with 1
    weights = np.ones((size, size))
    # Normalize weights
    weights = weights / np.sum(weights)
    # Ensure weights array is 2D
    if im_gaus.ndim == 2:
        weights = weights
    elif im_gaus.ndim == 3:
        weights = weights[:, :, np.newaxis]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im_gaus, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    ax[1].imshow(correlate(im_gaus, weights), cmap='gray')
    ax[1].set_title('Filtered image')
    ax[1].axis('off')
    plt.show()
#showImageWithAndWithoutFilter()

#Exercise 4
#For size 5 a lot of noise is removed
#For size 10 thie image starts to get blurry and the edges are not as sharp
#For size 20 the image is very blurry
def applyMedianFilter():
    sizes = [5, 10, 20]
    fig, ax = plt.subplots(1, len(sizes) + 1, figsize=(15, 5))
    ax[0].imshow(im_gaus, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    for i, size in enumerate(sizes):
        footprint = np.ones((size, size))
        if im_gaus.ndim == 2:
            med_img = median(im_gaus, footprint)
        elif im_gaus.ndim == 3:
            med_img = median(im_gaus, np.repeat(footprint[:, :, np.newaxis], im_gaus.shape[2], axis=2))
        ax[i + 1].imshow(med_img, cmap='gray')
        ax[i + 1].set_title(f'Median filter size {size}')
        ax[i + 1].axis('off')
    
    plt.show()
#applyMedianFilter()

#Exercise 5
# For the mean filter the image is blurred
# For the median filter the noise is removed way better
def applyFiltersOnSaltPepper():
    sizes = [3, 5, 7]
    fig, ax = plt.subplots(2, len(sizes) + 1, figsize=(15, 10))
    
    ax[0, 0].imshow(im_salt, cmap='gray')
    ax[0, 0].set_title('Original image')
    ax[0, 0].axis('off')
    
    ax[1, 0].imshow(im_salt, cmap='gray')
    ax[1, 0].set_title('Original image')
    ax[1, 0].axis('off')
    
    for i, size in enumerate(sizes):
        # Mean filter
        mean_weights = np.ones((size, size)) / (size * size)
        if im_salt.ndim == 2:
            mean_filtered_img = correlate(im_salt, mean_weights)
        elif im_salt.ndim == 3:
            mean_filtered_img = correlate(im_salt, mean_weights[:, :, np.newaxis])
        ax[0, i + 1].imshow(mean_filtered_img, cmap='gray')
        ax[0, i + 1].set_title(f'Mean filter size {size}')
        ax[0, i + 1].axis('off')
        
        # Median filter
        footprint = np.ones((size, size))
        if im_salt.ndim == 2:
            median_filtered_img = median(im_salt, footprint)
        elif im_salt.ndim == 3:
            median_filtered_img = median(im_salt, np.repeat(footprint[:, :, np.newaxis], im_salt.shape[2], axis=2))
        ax[1, i + 1].imshow(median_filtered_img, cmap='gray')
        ax[1, i + 1].set_title(f'Median filter size {size}')
        ax[1, i + 1].axis('off')
    
    plt.show()
#applyFiltersOnSaltPepper()


# Exercise 6
# The edges of the image are more visible but still gets blurry with high sigma
def applyGaussianFilter():
    sigmas = [1, 2, 3]
    fig, ax = plt.subplots(1, len(sigmas) + 1, figsize=(15, 5))
    
    ax[0].imshow(im_gaus, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        gauss_img = gaussian(im_gaus, sigma=sigma)
        ax[i + 1].imshow(gauss_img, cmap='gray')
        ax[i + 1].set_title(f'Gaussian filter sigma {sigma}')
        ax[i + 1].axis('off')
    
    plt.show()
#applyGaussianFilter()


# While the mean filter blurs the image, the gaussian filter keeps the edges sharp and to some extend removes the noise
# The gaussian filter better highlights dark areas 
def applyFiltersOnCarImage():
    car_image_path = in_dir + "car.png"
    car_image = io.imread(car_image_path)
    car_image_gray = color.rgb2gray(car_image)
    
    # Apply median filter with large kernel sizes
    median_sizes = [15, 20, 25]
    fig, ax = plt.subplots(1, len(median_sizes) + 1, figsize=(20, 5))
    ax[0].imshow(car_image_gray, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    for i, size in enumerate(median_sizes):
        footprint = np.ones((size, size))
        median_filtered_img = median(car_image_gray, footprint)
        ax[i + 1].imshow(median_filtered_img, cmap='gray')
        ax[i + 1].set_title(f'Median filter size {size}')
        ax[i + 1].axis('off')
    
    # Apply Gaussian filter with large sigma values
    gaussian_sigmas = [5, 10, 15]
    fig, ax = plt.subplots(1, len(gaussian_sigmas) + 1, figsize=(20, 5))
    ax[0].imshow(car_image_gray, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    for i, sigma in enumerate(gaussian_sigmas):
        gauss_img = gaussian(car_image_gray, sigma=sigma)
        ax[i + 1].imshow(gauss_img, cmap='gray')
        ax[i + 1].set_title(f'Gaussian filter sigma {sigma}')
        ax[i + 1].axis('off')
    
    plt.show()

#applyFiltersOnCarImage()

# Prewitt filters highlights the edges of the image showing where we transition from dark to light
# The horizontal filter highlights the vertical edges and the vertical filter highlights the horizontal edges
def applyPrewittFilters():
    donald_image_path = in_dir + "donald_1.png"
    donald_image = io.imread(donald_image_path, as_gray=True)
    
    prewitt_h_img = prewitt_h(donald_image)
    prewitt_v_img = prewitt_v(donald_image)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(donald_image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    ax[1].imshow(prewitt_h_img, cmap='gray', vmin=-1, vmax=1)
    ax[1].set_title('Prewitt Horizontal')
    ax[1].axis('off')
    
    ax[2].imshow(prewitt_v_img, cmap='gray', vmin=-1, vmax=1)
    ax[2].set_title('Prewitt Vertical')
    ax[2].axis('off')
    
    plt.show()

#applyPrewittFilters()

#Exercise 9
def applyPrewittFilter():
    # The Prewitt filter highlights the edges of the image, showing where there is a transition from dark to light. Seemingly by combining the horizontal and vertical filters, the edges are highlighted in both directions.
    donald_image_path = in_dir + "donald_1.png"
    donald_image = io.imread(donald_image_path, as_gray=True)
    prewitt_img = prewitt(donald_image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(donald_image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(prewitt_img, cmap='gray', vmin=-1, vmax=1)
    ax[1].set_title('Prewitt Filter')
    ax[1].axis('off')

    plt.show()
#applyPrewittFilter()

#Exercise 10
# Mean gives better definition of the edges while the gaussian gives thicker lines
def detectEdges():
    elbow_image_path = in_dir + "ElbowCTSlice.png"
    elbow_image = io.imread(elbow_image_path, as_gray=True)
    
    size = 5

    # Apply Gaussian filter
    sigma = 2
    gauss_img = gaussian(elbow_image, sigma=sigma)

    footprint = np.ones((size, size))
    median_filtered_img = median(elbow_image, footprint)


    # Apply Prewitt filter to get gradients
    prewitt_mean_img = prewitt(median_filtered_img)
    prewitt_gauss_img = prewitt(gauss_img)
    
    # Use Otsu's thresholding method
    T_mean = threshold_otsu(prewitt_mean_img)
    T_gauss = threshold_otsu(gauss_img)
    
    # Apply threshold to get binary image
    binary_mean_img = prewitt_mean_img > T_mean
    binary_gaus_img = prewitt_gauss_img > T_gauss
    
    # Display results for mean
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0,0].imshow(elbow_image, cmap='gray')
    ax[0,0].set_title('Original image')
    ax[0,0].axis('off')
    
    ax[1,0].imshow(prewitt_mean_img, cmap='gray', vmin=-1, vmax=1)
    ax[1,0].set_title('Prewitt Filter')
    ax[1,0].axis('off')
    
    ax[1,2].imshow(binary_mean_img, cmap='gray')
    ax[1,2].set_title('Binary Image')
    ax[1,2].axis('off')

    
    # Display results for guassian
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0,1].imshow(elbow_image, cmap='gray')
    ax[0,1].set_title('Original image')
    ax[0,1].axis('off')
    
    ax[1,1].imshow(prewitt_gauss_img, cmap='gray', vmin=-1, vmax=1)
    ax[1,1].set_title('Gaussian Filter')
    ax[1,1].axis('off')
    
    ax[1,2].imshow(binary_gaus_img, cmap='gray')
    ax[1,2].set_title('Binary Image')
    ax[1,2].axis('off')
    
    plt.show()

def main():
    print("Running Exercise 1:")
    reflectAndConstant()
    
    print("\nRunning Exercise 2:")
    showImageWithAndWithoutFilter()
    
    print("\nRunning Exercise 3:")
    applyMedianFilter()
    
    print("\nRunning Exercise 4:")
    applyFiltersOnSaltPepper()
    
    print("\nRunning Exercise 5:")
    applyGaussianFilter()
    
    print("\nRunning Exercise 6:")
    applyFiltersOnCarImage()
    
    print("\nRunning Exercise 7:")
    applyPrewittFilters()
    
    print("\nRunning Exercise 8:")
    applyPrewittFilter()
    
    print("\nRunning Exercise 9:")
    detectEdges()

if __name__ == "__main__":
    main()

#detectEdges()