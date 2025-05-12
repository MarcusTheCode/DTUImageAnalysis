import numpy as np
from scipy.ndimage import correlate
from skimage import io, color
from skimage.filters import prewitt
import matplotlib.pyplot as plt
from combined import (
    apply_mean_filter, apply_median_filter, apply_gaussian_filter,
    apply_prewitt_filters, detect_edges, show_comparison, convert_gray_to_rgb
)

# Constants
data_dir = "data/filter/"
input_img = np.arange(25).reshape(5, 5)
weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])

def exercise_1():
    res_img = correlate(input_img, weights)
    print("Exercise 1 - Correlation Result at (3, 3):", res_img[3, 3])

def exercise_2():
    reflect = correlate(input_img, weights, mode='reflect')
    constant = correlate(input_img, weights, mode='constant', cval=10)
    print("Exercise 2 - Reflect mode:\n", reflect)
    print("Exercise 2 - Constant mode:\n", constant)

def exercise_3():
    image = io.imread(data_dir + "Gaussian.png")
    blurred = apply_mean_filter(image, size=10)
    show_comparison(image, blurred, "Mean Filter (size 10)")

def exercise_4():
    image = io.imread(data_dir + "Gaussian.png")
    sizes = [5, 10, 20]
    for size in sizes:
        filtered = apply_median_filter(image, size)
        show_comparison(image, filtered, f"Median Filter (size {size})")

def exercise_5():
    image = io.imread(data_dir + "SaltPepper.png")
    sizes = [3, 5, 7]
    for size in sizes:
        mean_filtered = apply_mean_filter(image, size)
        median_filtered = apply_median_filter(image, size)
        show_comparison(image, mean_filtered, f"Mean Filter (size {size})")
        show_comparison(image, median_filtered, f"Median Filter (size {size})")

def exercise_6():
    image = io.imread(data_dir + "Gaussian.png")
    sigmas = [1, 2, 3]
    for sigma in sigmas:
        blurred = apply_gaussian_filter(image, sigma)
        show_comparison(image, blurred, f"Gaussian Filter (sigma {sigma})")

def exercise_7():
    image = io.imread(data_dir + "car.png")
    gray = color.rgb2gray(image)
    show_comparison(image, gray, "Original Image")
    for size in [15, 20, 25]:
        filtered = apply_median_filter(gray, size)
        show_comparison(gray, filtered, f"Median Filter (size {size})")
    for sigma in [5, 10, 15]:
        filtered = apply_gaussian_filter(gray, sigma)
        show_comparison(gray, filtered, f"Gaussian Filter (sigma {sigma})")

def exercise_8():
    image = io.imread(data_dir + "donald_1.png")
    gray = color.rgb2gray(image)
    h, v, _ = apply_prewitt_filters(gray)
    show_comparison(gray, h, "Prewitt Horizontal")
    show_comparison(gray, v, "Prewitt Vertical")

def exercise_9():
    image = io.imread(data_dir + "donald_1.png")
    gray = color.rgb2gray(image)
    combined = prewitt(gray)
    show_comparison(gray, combined, "Prewitt Combined")

def exercise_10():
    image = io.imread(data_dir + "ElbowCTSlice.png")
    gray = color.rgb2gray(image)
    for ftype in ['median', 'gaussian']:
        edges, threshold = detect_edges(gray, filter_type=ftype, param=5 if ftype == 'median' else 2)
        show_comparison(gray, edges, f"Edges using {ftype} filter")

# Entry point
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
    exercise_7()
    exercise_8()
    exercise_9()
    exercise_10()