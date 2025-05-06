from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib

# Read the image into im_org
#model_cat = io.imread('data/ModelCat.jpg')
#missing_cat = io.imread('data/MissingCat.jpg')

#model_cat_cat = io.imread('data/ModelCat.jpg.cat')
#missing_cat_cat = io.imread('data/MissingCat.jpg.cat')

def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm

def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None

def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)

def preprocess_one_cat():
    src = "data/MissingCat"
    dst = "data/ModelCat"
    out = "data/MissingCatProcessed.jpg"

    src_lm = read_landmark_file(f"{src}.jpg.cat")
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")

    src_img = io.imread(f"{src}.jpg")
    dst_img = io.imread(f"{dst}.jpg")

    src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if src_proc is None:
        return

    io.imsave(out, src_proc)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
    ax[1].imshow(dst_img)
    ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
    ax[2].imshow(src_proc)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out



#preprocess_all_cats("data/", "data/")
def collect_image_data_matrix(preprocessed_dir):
    image_files = glob.glob(preprocessed_dir + "*_preprocessed.jpg")
    n_samples = len(image_files)
    
    if n_samples == 0:
        print("No preprocessed images found.")
        return None

    # Read the first image to get dimensions
    first_image = io.imread(image_files[0])
    height, width, channels = first_image.shape
    n_features = height * width * channels

    # Initialize the data matrix
    data_matrix = np.zeros((n_samples, n_features))

    # Populate the data matrix
    for idx, image_file in enumerate(image_files):
        image = io.imread(image_file)
        flat_img = image.flatten()
        data_matrix[idx, :] = flat_img

    # Compute the average cat
    mean_cat_vector = np.mean(data_matrix, axis=0)

    # Create an image from the mean cat vector
    height, width, channels = first_image.shape
    mean_cat_image = create_u_byte_image_from_vector(mean_cat_vector, height, width, channels)

    # Save and display the mean cat image
    io.imsave("data/MeanCat.jpg", mean_cat_image)

    plt.imshow(mean_cat_image)
    plt.axis('off')
    plt.title("Mean Cat")
    plt.show()

    return data_matrix

# Example usage
data_matrix = collect_image_data_matrix("data/")
print(f"Data matrix shape: {data_matrix.shape}")


def find_similar_cat(missing_cat_image, data_matrix, preprocessed_dir):
    """
    Find a cat that looks like the missing cat using sum-of-squared differences (SSD)
    :param missing_cat_image: Path to the missing cat image
    :param data_matrix: Data matrix of preprocessed cat images
    :param preprocessed_dir: Directory of preprocessed cat images
    :return: Path to the most similar cat image
    """
    # Read and preprocess the missing cat image
    missing_cat_img = io.imread(missing_cat_image)
    missing_cat_flat = missing_cat_img.flatten()

    # Compute SSD between the missing cat and each cat in the training set
    ssd = np.sum((data_matrix - missing_cat_flat) ** 2, axis=1)

    # Find the index of the most similar cat
    most_similar_idx = np.argmin(ssd)

    # Get the corresponding file name of the most similar cat
    image_files = glob.glob(preprocessed_dir + "*_preprocessed.jpg")
    most_similar_cat = image_files[most_similar_idx]

    # Display the missing cat and the most similar cat
    most_similar_cat_img = io.imread(most_similar_cat)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].imshow(missing_cat_img)
    ax[0].set_title("Missing Cat")
    ax[0].axis('off')
    ax[1].imshow(most_similar_cat_img)
    ax[1].set_title("Most Similar Cat")
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

    return most_similar_cat

# Example usage
#missing_cat_image = "data/MissingCatProcessed.jpg"
#most_similar_cat = find_similar_cat(missing_cat_image, data_matrix, "data/")
#print(f"Most similar cat found at: {most_similar_cat}")

#preprocess_one_cat()
