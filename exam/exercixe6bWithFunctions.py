import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread_collection
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import os
from combined import plot_eigenfaces, reconstruct_image, perform_pca, load_mat_images

def run_exercise1(image_dir, data_matrix, image_shape):
    print("Exercise 1: Displaying the first image")
    plt.imshow(data_matrix[0].reshape(image_shape), cmap='gray')
    plt.title("First Image")
    plt.axis('off')
    plt.show()


def run_exercise2(data_matrix):
    print("Exercise 2: Mean face")
    mean_face = np.mean(data_matrix, axis=0)
    return mean_face


def run_exercise3(mean_face, image_shape):
    print("Exercise 3: Plot mean face")
    plt.imshow(mean_face.reshape(image_shape), cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')
    plt.show()


def run_exercise4(data_matrix, mean_face):
    print("Exercise 4: Center the data")
    return data_matrix - mean_face


def run_exercise5(data_matrix, n_components=10):
    print("Exercise 5: Perform PCA")
    return perform_pca(data_matrix, n_components=n_components)


def run_exercise6(pca, image_shape):
    print("Exercise 6: Display eigenfaces")
    plot_eigenfaces(pca, image_shape, num_faces=5)


def run_exercise7(pca, data_matrix):
    print("Exercise 7: PCA transform")
    return pca.transform(data_matrix)


def run_exercise8(pca, image_vector):
    print("Exercise 8: Reconstruct image")
    return pca.inverse_transform(pca.transform([image_vector]))


def run_exercise9(image_vector, image_shape):
    print("Exercise 9: Show original image")
    plt.imshow(image_vector.reshape(image_shape), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()


def run_exercise10(reconstructed_image, image_shape):
    print("Exercise 10: Show reconstructed image")
    plt.imshow(reconstructed_image.reshape(image_shape), cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()


def run_exercise11():
    print("Exercise 11: Full PCA workflow")
    in_dir = 'data/'
    in_file = 'ex6_ImagData2Load.mat'
    ImgT1, ImgT2, ROI_GM, ROI_WM = load_mat_images(in_dir + in_file)

    run_exercise1(image_dir, data_matrix, image_shape)
    mean_face = run_exercise2(data_matrix)
    run_exercise3(mean_face, image_shape)
    centered_data = run_exercise4(data_matrix, mean_face)
    pca = run_exercise5(centered_data, n_components=10)
    run_exercise6(pca, image_shape)
    transformed_data = run_exercise7(pca, centered_data)
    reconstructed = run_exercise8(pca, centered_data[0])
    run_exercise9(centered_data[0], image_shape)
    run_exercise10(reconstructed, image_shape)


if __name__ == "__main__":
    run_exercise11()
