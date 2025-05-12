import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from combined import apply_mean_filter, apply_median_filter, apply_gaussian_filter, apply_prewitt_filters, detect_edges


# Exercise 1: Correlation Center Value
def exercise_1():
    input_img = np.arange(25).reshape(5, 5)
    weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    from scipy.ndimage import correlate
    res_img = correlate(input_img, weights)
    print("Value at (3,3):", res_img[3, 3])


# Exercise 2: Border Modes Comparison
def exercise_2():
    from scipy.ndimage import correlate
    input_img = np.arange(25).reshape(5, 5)
    weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    reflect_result = correlate(input_img, weights, mode="reflect")
    constant_result = correlate(input_img, weights, mode="constant", cval=10)
    print("Reflect mode:\n", reflect_result)
    print("Constant mode:\n", constant_result)


# Exercise 3: Mean Filter
def exercise_3():
    im_org = io.imread("Gaussian.png", as_gray=True)
    filtered = apply_mean_filter(im_org, size=5)
    plt.subplot(1, 2, 1)
    plt.imshow(im_org, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Mean Filtered (5x5)")
    plt.show()


# Exercise 4: Median Filter
def exercise_4():
    im_org = io.imread("Gaussian.png", as_gray=True)
    med_img = apply_median_filter(im_org, size=5)
    plt.subplot(1, 2, 1)
    plt.imshow(im_org, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(med_img, cmap='gray')
    plt.title("Median Filtered (5x5)")
    plt.show()


# Exercise 5: Salt-and-Pepper Filtering
def exercise_5():
    img = io.imread("SaltPepper.png", as_gray=True)
    mean_filtered = apply_mean_filter(img, size=5)
    median_filtered = apply_median_filter(img, size=5)
    plt.subplot(1, 3, 1)
    plt.imshow(imgimport numpy as np
import matplotlib.pyplot as plt
from skimage import io
from combined import apply_mean_filter, apply_median_filter, apply_gaussian_filter, apply_prewitt_filters, detect_edges


# Exercise 1: Correlation Center Value
def exercise_1():
    input_img = np.arange(25).reshape(5, 5)
    weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    from scipy.ndimage import correlate
    res_img = correlate(input_img, weights)
    print("Value at (3,3):", res_img[3, 3])


# Exercise 2: Border Modes Comparison
def exercise_2():
    from scipy.ndimage import correlate
    input_img = np.arange(25).reshape(5, 5)
    weights = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    reflect_result = correlate(input_img, weights, mode="reflect")
    constant_result = correlate(input_img, weights, mode="constant", cval=10)
    print("Reflect mode:\n", reflect_result)
    print("Constant mode:\n", constant_result)


# Exercise 3: Mean Filter
def exercise_3():
    im_org = io.imread("Gaussian.png", as_gray=True)
    filtered = apply_mean_filter(im_org, size=5)
    plt.subplot(1, 2, 1)
    plt.imshow(im_org, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Mean Filtered (5x5)")
    plt.show()


# Exercise 4: Median Filter
def exercise_4():
    im_org = io.imread("Gaussian.png", as_gray=True)
    med_img = apply_median_filter(im_org, size=5)
    plt.subplot(1, 2, 1)
    plt.imshow(im_org, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(med_img, cmap='gray')
    plt.title("Median Filtered (5x5)")
    plt.show()


# Exercise 5: Salt-and-Pepper Filtering
def exercise_5():
    img = io.imread("SaltPepper.png", as_gray=True)
    mean_filtered = apply_mean_filter(img, size=5)
    median_filtered = apply_median_filter(img, size=5)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Noisy Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mean_filtered, cmap='gray')
    plt.title("Mean Filter")
    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median Filter")
    plt.show()


# Exercise 6: Gaussian Filter
def exercise_6():
    im_org = io.imread("Gaussian.png", as_gray=True)
    gauss_img = apply_gaussian_filter(im_org, sigma=1)
    plt.imshow(gauss_img, cmap='gray')
    plt.title("Gaussian σ=1")
    plt.show()


# Exercise 7: Free Image Filtering
def exercise_7():
    img = io.imread("car.png", as_gray=True)
    med_filtered = apply_median_filter(img, size=15)
    gauss_filtered = apply_gaussian_filter(img, sigma=5)
    plt.imshow(med_filtered, cmap='gray')
    plt.title("Median Filter (15x15)")
    plt.show()
    plt.imshow(gauss_filtered, cmap='gray')
    plt.title("Gaussian σ=5")
    plt.show()


# Exercise 8: Prewitt H/V Filters
def exercise_8():
    img = io.imread("donald_1.png", as_gray=True)
    ph, pv, _ = apply_prewitt_filters(img)
    plt.subplot(1, 2, 1)
    plt.imshow(ph, cmap='gray')
    plt.title("Prewitt Horizontal")
    plt.subplot(1, 2, 2)
    plt.imshow(pv, cmap='gray')
    plt.title("Prewitt Vertical")
    plt.show()


# Exercise 9: Prewitt Magnitude
def exercise_9():
    img = io.imread("donald_1.png", as_gray=True)
    _, _, pm = apply_prewitt_filters(img)
    plt.imshow(pm, cmap='gray')
    plt.title("Prewitt Magnitude")
    plt.show()


# Exercise 10: Edge Detection in CT
def exercise_10():
    img = io.imread("ElbowCTSlice.png", as_gray=True)
    binary, threshold = detect_edges(img, filter_type='gaussian', param=2)
    plt.imshow(binary, cmap='gray')
    plt.title(f"Detected Edges (Otsu threshold: {threshold:.4f})")
    plt.show()


# Exercise 11 & 12: Webcam Video Filtering Example
# These would be implemented in a script using OpenCV, similar to:
# cap = cv2.VideoCapture(0)
# while True: ...
# For real-time application, this is left out for a script context.
, cmap='gray')
    plt.title("Noisy Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mean_filtered, cmap='gray')
    plt.title("Mean Filter")
    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median Filter")
    plt.show()


# Exercise 6: Gaussian Filter
def exercise_6():
    im_org = io.imread("Gaussian.png", as_gray=True)
    gauss_img = apply_gaussian_filter(im_org, sigma=1)
    plt.imshow(gauss_img, cmap='gray')
    plt.title("Gaussian σ=1")
    plt.show()


# Exercise 7: Free Image Filtering
def exercise_7():
    img = io.imread("car.png", as_gray=True)
    med_filtered = apply_median_filter(img, size=15)
    gauss_filtered = apply_gaussian_filter(img, sigma=5)
    plt.imshow(med_filtered, cmap='gray')
    plt.title("Median Filter (15x15)")
    plt.show()
    plt.imshow(gauss_filtered, cmap='gray')
    plt.title("Gaussian σ=5")
    plt.show()


# Exercise 8: Prewitt H/V Filters
def exercise_8():
    img = io.imread("donald_1.png", as_gray=True)
    ph, pv, _ = apply_prewitt_filters(img)
    plt.subplot(1, 2, 1)
    plt.imshow(ph, cmap='gray')
    plt.title("Prewitt Horizontal")
    plt.subplot(1, 2, 2)
    plt.imshow(pv, cmap='gray')
    plt.title("Prewitt Vertical")
    plt.show()


# Exercise 9: Prewitt Magnitude
def exercise_9():
    img = io.imread("donald_1.png", as_gray=True)
    _, _, pm = apply_prewitt_filters(img)
    plt.imshow(pm, cmap='gray')
    plt.title("Prewitt Magnitude")
    plt.show()


# Exercise 10: Edge Detection in CT
def exercise_10():
    img = io.imread("ElbowCTSlice.png", as_gray=True)
    binary, threshold = detect_edges(img, filter_type='gaussian', param=2)
    plt.imshow(binary, cmap='gray')
    plt.title(f"Detected Edges (Otsu threshold: {threshold:.4f})")
    plt.show()


# Exercise 11 & 12: Webcam Video Filtering Example
# These would be implemented in a script using OpenCV, similar to:
# cap = cv2.VideoCapture(0)
# while True: ...
# For real-time application, this is left out for a script context.
