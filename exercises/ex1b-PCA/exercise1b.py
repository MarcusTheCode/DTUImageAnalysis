from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import seaborn as sns
import pandas as pd
from sklearn import decomposition

# Directory containing data and images
in_dir = "data/"

# X-ray image
txt_name = "irisdata.txt"

#Exercise 1
def readIrisData():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    # x is a matrix with 50 rows and 4 columns
    x = iris_data[0:50, 0:4]
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")
#readIrisData()

#Exercise 2
def createIndividualVectors():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    sep_l = x[:, 0]
    sep_w = x[:, 1]
    pet_l = x[:, 2]
    pet_w = x[:, 3]

    # Variance of the individual vectors
    var_sep_l = sep_l.var(ddof=1)
    var_sep_w = sep_w.var(ddof=1)
    var_pet_l = pet_l.var(ddof=1)
    var_pet_w = pet_w.var(ddof=1)


    print(f"Sepal length variance: {var_sep_l}")
    print(f"Sepal width variance: {var_sep_w}")
    print(f"Petal length variance: {var_pet_l}")
    print(f"Petal width variance: {var_pet_w}")
#createIndividualVectors()

def covariance(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

#Exercise 3
# There is little relation between the sepal length and the petal length
# There is high relation between the sepal width and the sepal length
def findCovarianceBetweenPedals():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    sep_l = x[:, 0]
    sep_w = x[:, 1]
    pet_l = x[:, 2]
    pet_w = x[:, 3]

    print(covariance(pet_l, sep_l))
    print(covariance(sep_w, sep_l))
#findCovarianceBetweenPedals()

#Exercise 4
# You can se that the sepal length and sepal width are highly correlated, while the sepal length and petal length are not
def parPlotExmaple():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    df = pd.DataFrame(x, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
    sns.pairplot(df)
    plt.show()
#parPlotExmaple()

#Exercise 5
def PCAAnalysis():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    # Step 1: Subtract the mean
    mn = np.mean(x, axis=0)
    data = x - mn

    # Step 2: Compute covariance matrix manually
    N = data.shape[0]  # Number of samples
    cov_matrix_manual = (1 / (N - 1)) * np.matmul(data.T, data)

    # Step 3: Compute covariance matrix using numpy function
    cov_matrix_numpy = np.cov(x, rowvar=False)

    # Step 4: Verify if they give the same result
    print("Manually Computed Covariance Matrix:\n", cov_matrix_manual)
    print("NumPy Computed Covariance Matrix:\n", cov_matrix_numpy)
    print("Are they equal?", np.allclose(cov_matrix_manual, cov_matrix_numpy))
#PCAAnalysis()

#Exercise 6
def eigenvectorAnalysis():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    # Step 1: Subtract the mean
    mn = np.mean(x, axis=0)
    data = x - mn

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    print("Eigenvalues:\n", eig_values)
    print("Eigenvectors:\n", eig_vectors)
#eigenvectorAnalysis()

#Exercise 7
def explainedVariation():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    # Step 1: Subtract the mean
    mn = np.mean(x, axis=0)
    data = x - mn

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov_matrix)
    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel("Principal component")
    plt.ylabel("Percent explained variance")
    plt.ylim([0, 100])
    plt.show()
#explainedVariation()

#Exercise 8
def projectData():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    # Step 1: Subtract the mean
    mn = np.mean(x, axis=0)
    data = x - mn

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov_matrix)

    # Step 4: Project the data
    proj = vectors.T.dot(data.T)
    plt.plot(proj)
    plt.xlabel("Principal component")
    plt.ylabel("Percent explained variance")
    plt.show()
#projectData()

#Exercise 9
def pcaComputed():
    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = iris_data[0:50, 0:4]
    pca = decomposition.PCA()
    pca.fit(x)
    values_pca = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    vectors_pca = pca.components_

    data_transform = pca.transform(x)
