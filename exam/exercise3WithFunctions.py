from skimage import color, io, img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

# Import from combined.py
from combined import (
    show_comparison,
    apply_gaussian_filter,
    apply_median_filter,
    mask_above_threshold,
    set_pixels_with_mask,
    show_min_max,
    show_image_with_histogram
)

def load_image(filename):
    return io.imread(DATA_DIR + filename)

# Set base data directory
DATA_DIR = "data/pixelwiseOperations/"

def exercise1():
    image = load_image("vertebra.png")
    show_image_with_histogram(image, "Vertebra Image")

def exercise2():
    image = load_image("vertebra.png")
    show_min_max(image, "Original")

def exercise3():
    image = load_image("vertebra.png")
    image_float = img_as_float(image)
    show_min_max(image, "Original")
    show_min_max(image_float, "Float")

def exercise4():
    image = load_image("vertebra.png")
    image_ubyte = img_as_ubyte(img_as_float(image))
    show_min_max(image_ubyte, "UByte")

def histogram_stretch(img_in):
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    img_out = (img_float - min_val) / (max_val - min_val)
    return img_as_ubyte(img_out)

def exercise5():
    image = load_image("vertebra.png")
    stretched = histogram_stretch(image)
    show_image_with_histogram(stretched, "Histogram Stretched")

def exercise6():
    image = load_image("vertebra.png")
    stretched = histogram_stretch(image)
    show_image_with_histogram(image, "Original Image")
    show_image_with_histogram(stretched, "Stretched Image")

def gamma_map(img, gamma):
    img_float = img_as_float(img)
    img_gamma = np.power(img_float, gamma)
    return img_as_ubyte(img_gamma)

def exercise7():
    image = load_image("vertebra.png")
    for gamma in [0.5, 2.0]:
        corrected = gamma_map(image, gamma)
        show_image_with_histogram(corrected, f"Gamma {gamma}")

def exercise8():
    image = load_image("vertebra.png")
    for threshold in [50, 100, 150, 200]:
        binary = set_pixels_with_mask(
            np.zeros_like(image, dtype=np.uint8),
            mask_above_threshold(image, threshold),
            255,
        )
        show_image_with_histogram(binary, f"Threshold {threshold}")

def exercise9():
    image = load_image("vertebra.png")
    img_ubyte = img_as_ubyte(image)
    threshold = threshold_otsu(img_ubyte)
    binary = set_pixels_with_mask(
        np.zeros_like(img_ubyte, dtype=np.uint8),
        img_ubyte > threshold,
        255,
    )
    show_image_with_histogram(binary, f"Otsu Threshold {threshold}")

def exercise10():
    image = load_image("dark_background.png")
    gray = color.rgb2gray(image)
    threshold = threshold_otsu(gray)
    binary = set_pixels_with_mask(
        np.zeros_like(gray, dtype=np.uint8),
        gray > threshold,
        255,
    )
    show_image_with_histogram(binary, f"Dark BG Otsu Threshold {threshold}")

def detect_dtu_signs_rgb(img, color_name="blue"):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if color_name == "blue":
        return ((r < 10) & (g > 85) & (g < 105) & (b > 180) & (b < 200)).astype(np.uint8) * 255
    elif color_name == "red":
        return ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8) * 255
    else:
        raise ValueError("Unsupported color")

def exercise11():
    img = load_image("DTUSigns2.jpg")
    for c in ["blue", "red"]:
        mask = detect_dtu_signs_rgb(img, color_name=c)
        show_comparison(img, mask, f"Detected {c} sign")

def detect_dtu_signs_hsv(img, sign_color):
    hsv = color.rgb2hsv(img)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if sign_color == "blue":
        return ((h > 0.5) & (h < 0.7) & (s > 0.4) & (v > 0.2)).astype(np.uint8) * 255
    elif sign_color == "red":
        return (((h < 0.05) | (h > 0.95)) & (s > 0.4) & (v > 0.2)).astype(np.uint8) * 255
    raise ValueError("Unsupported sign_color")

def exercise12():
    img = load_image("DTUSigns2.jpg")
    for c in ["blue", "red"]:
        mask = detect_dtu_signs_hsv(img, c)
        show_comparison(img, mask, f"HSV Detected {c} sign")

if __name__ == "__main__":
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    exercise5()
    exercise6()
    exercise7()
    exercise8()
    exercise9()
    exercise10()
    exercise11()
    exercise12()
