from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, img_as_ubyte
from combined import (convert_to_grayscale, threshold_image, remove_border_blobs, apply_closing,
                      label_image, visualize_labels, apply_opening, cell_counting_ex17, preprocess_to_binary,
                      solution_circularities, cell_counting, show_comparison, im2patch)

# Input image paths
data_path = './data/blob/'
lego_img_path = data_path + 'lego_4_small.png'
dapi_img_path = data_path + 'Sample E2 - U2OS DAPI channel.tiff'


def run_exercise1():
    img = io.imread(lego_img_path)
    gray = convert_to_grayscale(img)
    binary = threshold_image(gray, 0)
    show_comparison(img, binary, 'Exercise 1: Binary Image')
    return binary


def run_exercise2(binary):
    cleaned = remove_border_blobs(binary)
    show_comparison(binary, cleaned, 'Exercise 2: Border Removed')
    return cleaned


def run_exercise3(binary):
    im_process = remove_border_blobs(binary)
    im_process = apply_closing(im_process, radius=5)
    im_open = apply_opening(im_process, radius=5)
    show_comparison(binary, im_open, 'Exercise 2: Post Morphology')
    return im_open


def run_exercise4(binary):
    labeled = label_image(binary)
    return labeled


def run_exercise5(labeled, image):
    visual = visualize_labels(labeled)
    show_comparison(image, visual, 'Exercise 5: Label Visualization')
    return visual


def run_exercise6(labeled, image):
    region_props = measure.regionprops(labeled)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.title("Exercise 6: Histogram of BLOB Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.show()
    return region_props


def run_exercise7(props):
    print("Exercise 7 requires running the interactive script 'Ex5-BlobAnalysisInteractive.py'")


def run_exercise8():
    img_org = io.imread(dapi_img_path)
    img_small = img_org[700:1200, 900:1400]
    img_gray = img_as_ubyte(img_small)

    io.imshow(img_gray, vmin=0, vmax=150)
    plt.title('DAPI Stained U2OS cell nuclei')
    io.show()

    plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
    plt.title('Histogram of intensities (excluding 0)')
    io.show()

    binary = threshold_image(img_gray, 1)
    show_comparison(img_gray, binary, 'binary')
    return img_small, binary



def run_exercise9(img_gray, binary):
    img_c_b = remove_border_blobs(binary)
    label_img = measure.label(img_c_b)
    image_label_overlay = visualize_labels(label_img)
    show_comparison(img_gray, image_label_overlay, 'Found BLOBS')
    return label_img


def run_exercise10(label_img):
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()
    return region_props, areas

def run_exercise11(label_img, img_gray):
    min_area = 10
    max_area = 150
    region_props = measure.regionprops(label_img)
    label_img_filter = np.copy(label_img)

    for region in region_props:
        if region.area < min_area or region.area > max_area:
            for coords in region.coords:
                label_img_filter[coords[0], coords[1]] = 0

    i_area = label_img_filter > 0
    show_comparison(img_gray, i_area, 'Found nuclei based on area')
    return i_area


def run_exercise12(props, areas):
    perimeters = np.array([prop.perimeter for prop in props])
    fig, ax = plt.subplots(1,1)
    ax.plot(areas, perimeters, '.')
    ax.set_xlabel('Areas (px)')
    ax.set_ylabel('Perimeter (px)')
    plt.show()
    return perimeters


def run_exercise13(areas, perimeters, label_img, region_props, img_small):
    circs = solution_circularities(areas, perimeters)
    plt.hist(circs, bins=50)
    plt.show()

    min_circ = 0.7
    min_area = 10 
    max_area = 150

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        circ = solution_circularities(region.area, region.perimeter)
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area or circ < min_circ:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img_small, i_area, 'Found nuclei based on area and circularity')
    return circs


def run_exercise14(areas, circs):
    fig, ax = plt.subplots(1,1)
    ax.plot(areas, circs, '.')
    ax.set_xlabel('Areas (px)')
    ax.set_ylabel('Circularity')
    plt.show()


def run_exercise15():
    img_org = io.imread(data_path + 'Sample E2 - U2OS DAPI channel.tiff')
    img_gray = img_as_ubyte(img_org)
    patches = im2patch(img_gray, patch_size=[300,300])
    print(f'Number of patches {patches.shape[-1]}')

    for idx_patch in range(6):
        patch = patches[:,:,idx_patch]
        filt_label, n_nuclei = cell_counting(patch)
        show_comparison(patch, filt_label, f'Found nuclei: {n_nuclei}')


def run_exercise16():
    img_org = io.imread(data_path + 'Sample G1 - COS7 cells DAPI channel.tiff')
    img_gray = img_as_ubyte(img_org)

    patches = im2patch(img_gray, patch_size=[300,300])
    print(f'Number of patches {patches.shape[-1]}')

    for idx_patch in range(6):
        patch = patches[:,:,idx_patch]
        filt_label, n_nuclei = cell_counting(patch)
        show_comparison(patch, filt_label,  f'Found nuclei: {n_nuclei}')


def run_exercise17():
    img_org = io.imread(data_path + 'Sample G1 - COS7 cells DAPI channel.tiff')
    img_gray = img_as_ubyte(img_org)

    patches = im2patch(img_gray, patch_size=[300,300])
    print(f'Number of patches {patches.shape[-1]}')

    for idx_patch in range(6):
        patch = patches[:,:,idx_patch]
        filt_label, n_nuclei = cell_counting_ex17(patch, opening_sz = 3)
        show_comparison(patch, filt_label, f'Found nuclei: {n_nuclei}')


def main():
    img_org = io.imread(data_path + 'Sample E2 - U2OS DAPI channel.tiff')
    # slice to extract smaller image
    img_small = img_org[700:1200, 900:1400]
    binary1 = run_exercise1()
    binary2 = run_exercise2(binary1)
    binary3 = run_exercise3(binary2)
    labeled = run_exercise4(binary3)
    image = io.imread(lego_img_path)
    run_exercise5(labeled, image)
    props = run_exercise6(labeled, image)
    run_exercise7(props)

    img_gray, props8 = run_exercise8()
    label_img = run_exercise9(img_gray, props8)
    region_props, areas= run_exercise10(label_img)

    i_area = run_exercise11(label_img, img_gray)
    perimeters = run_exercise12(region_props, areas)
    circs = run_exercise13(areas, perimeters, label_img, region_props, img_small)
    run_exercise14(areas, circs)
    run_exercise15()
    run_exercise16()
    run_exercise17()


if __name__ == "__main__":
    main()