import numpy as np
from data.LDA import LDA
import matplotlib.pyplot as plt
import scipy.io as sio

in_dir = 'data/'
in_file = 'ex6_ImagData2Load.mat'
data = sio.loadmat(in_dir + in_file)
ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM'].astype(bool)
ROI_WM = data['ROI_WM'].astype(bool)

# Exercise 1

def exercise1():
    # Mask to remove background voxels
    mask = (ImgT1 > 0) & (ImgT2 > 0)
    ImgT1_masked = ImgT1[mask & (ImgT1 > 0)]
    ImgT2_masked = ImgT2[mask & (ImgT2 > 0)]
    # Collect all figures into one page
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # T1 and T2 images
    axes[0, 0].imshow(ImgT1, cmap='gray')
    axes[0, 0].set_title('T1 Image')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(ImgT2, cmap='gray')
    axes[0, 1].set_title('T2 Image')
    axes[0, 1].axis('off')

    # 1D Histograms
    axes[1, 0].hist(ImgT1_masked.ravel(), bins=50, color='blue', alpha=0.7)
    axes[1, 0].set_title('T1 Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 1].hist(ImgT2_masked.ravel(), bins=50, color='green', alpha=0.7)
    axes[1, 1].set_title('T2 Histogram')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')

    # 2D Histogram
    hist2d = axes[2, 0].hist2d(ImgT1_masked.ravel(), ImgT2_masked.ravel(), bins=50, cmap='viridis')
    plt.colorbar(hist2d[3], ax=axes[2, 0], label='Frequency')
    axes[2, 0].set_title('2D Histogram of T1 vs T2')
    axes[2, 0].set_xlabel('T1 Intensity')
    axes[2, 0].set_ylabel('T2 Intensity')

    # Scatter Plot
    axes[2, 1].scatter(ImgT1_masked.ravel(), ImgT2_masked.ravel(), s=1, alpha=0.5, color='purple')
    axes[2, 1].set_title('Scatter Plot of T1 vs T2')
    axes[2, 1].set_xlabel('T1 Intensity')
    axes[2, 1].set_ylabel('T2 Intensity')

    #plt.tight_layout()
    #plt.show()

# Q1 around 550
# Q2 Yes the scatter plot shows a clear seperation between the two tissues


def exercise2():
    # Place training examples into variables
    C1 = ROI_WM
    C2 = ROI_GM

    # Show manually expert drawings of the training examples
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(C1, cmap='gray')
    axes[0].set_title('Class 1 (WM) Training Examples')
    axes[0].axis('off')

    axes[1].imshow(C2, cmap='gray')
    axes[1].set_title('Class 2 (GM) Training Examples')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Q3: Does the ROI drawings look like what you expect from an expert?
    # No the drawings does not incapsulate the expected regions of interest
#exercise2()

#Exercise 3,4,5
def exercise3():
    qC1 = np.argwhere(ROI_WM)
    qC2 = np.argwhere(ROI_GM)

    # Extract training examples for each class
    training_examples_C1_T1 = ImgT1[ROI_WM]
    training_examples_C1_T2 = ImgT2[ROI_WM]
    training_examples_C2_T1 = ImgT1[ROI_GM]
    training_examples_C2_T2 = ImgT2[ROI_GM]

    # Create training data vector X
    X_C1 = np.column_stack((training_examples_C1_T1, training_examples_C1_T2))
    X_C2 = np.column_stack((training_examples_C2_T1, training_examples_C2_T2))
    X = np.vstack((X_C1, X_C2))

    # Create target class vector T
    T_C1 = np.zeros(len(training_examples_C1_T1), dtype=int)
    T_C2 = np.ones(len(training_examples_C2_T1), dtype=int)
    T = np.concatenate((T_C1, T_C2))

    # Print shapes to verify
    print(f"Shape of X: {X.shape}")
    print(f"Shape of T: {T.shape}")
    # Find indices of voxels belonging to each class

    # Print the number of training examples for each class
    print(f"Class 1 (WM) Training Examples: {len(training_examples_C1_T1)}")
    print(f"Class 2 (GM) Training Examples: {len(training_examples_C2_T1)}")

    # Scatter plot of training points
    
    plt.figure(figsize=(8, 6))
    plt.scatter(training_examples_C1_T1, training_examples_C1_T2, c='green', label='Class 1 (WM)', alpha=0.6)
    plt.scatter(training_examples_C2_T1, training_examples_C2_T2, c='black', label='Class 2 (GM)', alpha=0.6)
    plt.title('Scatter Plot of Training Points')
    plt.xlabel('T1 Intensity')
    plt.ylabel('T2 Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    W = LDA(X,T)
    print(W)
    Xall= np.c_[ImgT1.ravel(), ImgT2.ravel()]
    Y = np.c_[np.ones((len(Xall), 1)), Xall] @ W.T
    PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y),1)[:, np.newaxis], 0, 1)

    # Apply segmentation
    Class1 = np.where(PosteriorProb[:, 1] > 0.5, 1, 0).reshape(ImgT1.shape)
    Class2 = np.where(PosteriorProb[:, 1] <= 0.5, 1, 0).reshape(ImgT1.shape)

    # Visualize segmentation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(Class1, cmap='gray')
    plt.title('Class 1 (WM) Segmentation')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(Class2, cmap='gray')
    plt.title('Class 2 (GM) Segmentation')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Scatter plot of segmentation results
    plt.figure(figsize=(8, 6))
    plt.scatter(Xall[Class1.ravel() == 1, 0], Xall[Class1.ravel() == 1, 1], c='green', label='Class 1 (WM)', alpha=0.6)
    plt.scatter(Xall[Class2.ravel() == 1, 0], Xall[Class2.ravel() == 1, 1], c='black', label='Class 2 (GM)', alpha=0.6)
    plt.title('Scatter Plot of Segmentation Results')
    plt.xlabel('T1 Intensity')
    plt.ylabel('T2 Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Q6: Yes in the middle
"""
  Q7: Is the linear hyperplane positioned as you expected or would a non-linear hyperplane perform better?
If the scatter plot shows that the two classes (WM and GM) are linearly separable, then the linear hyperplane is positioned as expected. However, if there is significant overlap or non-linear separation between the two classes, a non-linear hyperplane (e.g., using a kernel-based method or a neural network) might perform better.

Q8: Would segmentation be as good as using a single image modality using thresholding?
No, segmentation using both T1 and T2 modalities is generally better than using a single modality with thresholding. Combining multiple modalities provides more information, allowing for better differentiation between tissue types, especially when their intensity ranges overlap in one modality.

Q9: From the scatter plot, does the segmentation result make sense? Are the two tissue types segmented correctly?
If the scatter plot shows two distinct clusters corresponding to WM and GM, and the segmentation aligns well with these clusters, then the segmentation results make sense, and the tissue types are segmented correctly. However, if there is significant misclassification or overlap, the segmentation may need improvement.

Q10: What are the limitations of the LDA method for this segmentation task?

"""

exercise3()

