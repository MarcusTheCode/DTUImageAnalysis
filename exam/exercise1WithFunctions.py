# Exercise 1 - Introduction to Image Analysis using combined.py

from skimage import io, color, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import combined  # Use module-level reference to avoid ImportError

# Directory setup
in_dir = "data/"

# Exercise 1–4: Basic image reading and properties
def exercise_1_to_4():
    im_org = io.imread(in_dir + "metacarpals.png")
    print("Shape:", im_org.shape)
    print("Pixel type:", im_org.dtype)
    combined.show_image(im_org, title="Metacarpal image")
    return im_org

# Exercise 5–6: Display with color maps
def exercise_5_to_6(im_org):
    for cmap in ["jet", "cool", "hot", "pink", "copper", "coolwarm", "cubehelix", "terrain"]:
        combined.apply_colormap(im_org, cmap, f"Metacarpal with colormap: {cmap}")

# Exercise 7: Auto gray level scaling
def exercise_7(im_org):
    combined.scale_image_display(im_org)

# Exercise 8–9: Histogram analysis
def exercise_8_to_9(im_org):
    counts, edges = combined.compute_histogram_data(im_org)
    combined.show_histogram(im_org)
    bin_no = int(np.argmax(counts))
    print("Most common intensity range:", edges[bin_no], "to", edges[bin_no + 1])

# Exercise 10: Pixel inspection
def exercise_10(im_org):
    combined.analyze_pixel(im_org, 110, 90)

# Exercise 11: Row manipulation
def exercise_11(im_org):
    im_temp = im_org.copy()
    im_temp[:30] = 0
    combined.show_image(im_temp)

# Exercise 12–13: Binary masking
def exercise_12_to_13(im_org):
    mask = combined.mask_above_threshold(im_org, 150)
    combined.show_image(mask)
    modified = combined.set_pixels_with_mask(im_org.copy(), mask, 255)
    combined.show_image(modified)

# Exercise 14–16: RGB manipulation
def exercise_14_to_16():
    im_rgb = io.imread(in_dir + "ardeche.jpg")
    print("RGB Shape:", im_rgb.shape)
    print("Type:", im_rgb.dtype)
    print("RGB at (110, 90):", im_rgb[110, 90])
    combined.extract_color_channels(im_rgb)
    r2 = im_rgb.shape[0] // 2
    im_rgb[:r2, :] = [0, 255, 0]
    combined.show_image(im_rgb)

# Exercise 17–19: Rescaling and resizing
def exercise_17_to_19():
    my_img = io.imread(in_dir + "P1010266.JPG")
    print("Shape before:", my_img.shape)
    rescaled = rescale(my_img, 0.25, anti_aliasing=True, channel_axis=2)
    print("After rescale:", rescaled.shape)
    print("Rescaled type:", rescaled.dtype)
    resized = resize(my_img, (my_img.shape[0] // 4, my_img.shape[1] // 6), anti_aliasing=True)
    print("After resize:", resized.shape)
    new_width = 400
    scale = new_width / my_img.shape[1]
    auto_resized = rescale(my_img, scale, anti_aliasing=True, channel_axis=2)

# Exercise 20–21: Histogram comparisons
def exercise_20_to_21():
    for img_name in ["dark.JPG", "bright.JPG", "darkBright.jpg"]:
        img = io.imread(in_dir + img_name)
        gray = color.rgb2gray(img)
        combined.show_histogram(gray)

# Exercise 22–23: Color channels
def exercise_22_to_23():
    im_dtu = io.imread(in_dir + "DTUSign1.jpg")
    combined.extract_color_channels(im_dtu)

# Exercise 24–26: Rectangle marking
def exercise_24_to_26():
    im_dtu = io.imread(in_dir + "DTUSign1.jpg")
    marked = combined.mark_region_black(im_dtu.copy(), (500, 1000, 800, 1500))
    io.imsave(in_dir + "DTUSign1-marked.jpg", marked)
    blue_box = combined.draw_colored_rectangle(im_dtu.copy(), (450, 550, 750, 1550), [0, 0, 255])
    io.imsave(in_dir + "DTUSign1-bluebox.jpg", blue_box)

# Exercise 27: Bone coloring
def exercise_27(im_org):
    gray = im_org if len(im_org.shape) == 2 else color.rgb2gray(im_org)
    rgb = combined.convert_gray_to_rgb(gray)
    mask = gray > 150
    highlighted = rgb.copy()
    highlighted[mask] = [0, 0, 255]  # Apply red highlight only to masked areas
    combined.show_image(highlighted)

# Exercise 28: Profile line analysis
def exercise_28(im_org):
    combined.profile_intensity(im_org, (342, 77), (320, 160))

# Exercise 29–30: DICOM
def exercise_29_to_30():
    image, metadata = combined.read_dicom_image(in_dir + "1-442.dcm")
    print(metadata)
    combined.show_image(image, vmin=-1000, vmax=1000, cmap='gray')

# Call functions for demonstration or testing (optional)
if __name__ == "__main__":
    im_org = exercise_1_to_4()
    exercise_5_to_6(im_org)
    exercise_7(im_org)
    exercise_8_to_9(im_org)
    exercise_10(im_org)
    exercise_11(im_org)
    exercise_12_to_13(im_org)
    exercise_14_to_16()
    exercise_17_to_19()
    exercise_20_to_21()
    exercise_22_to_23()
    exercise_24_to_26()
    exercise_27(im_org)
    exercise_28(im_org)
    exercise_29_to_30()