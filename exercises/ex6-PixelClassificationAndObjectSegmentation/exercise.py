from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


in_dir = "data/"
ct = dicom.read_file(in_dir + 'Training.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)
vmin, vmax = np.min(img), np.max(img)
io.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
io.show()

#Exercise 1
spleen_roi = io.imread(in_dir + 'SpleenROI.png')
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values = img[spleen_mask]
show_comparison(img, spleen_mask, 'Spleen ROI')


#Exercise 2
# Compute the average and standard deviation of the Hounsfield units in the spleen
spleen_mean = np.mean(spleen_values)
spleen_std = np.std(spleen_values)

print(f"Mean Hounsfield units in spleen: {spleen_mean}")
print(f"Standard deviation of Hounsfield units in spleen: {spleen_std}")

# Exercise 3
def plotPixelValueSpleen():
    # Plot a histogram of the pixel values of the spleen
    plt.hist(spleen_values, bins=50, density=True, alpha=0.6, color='g')

    # Fit a Gaussian curve to the histogram
    mu, std = norm.fit(spleen_values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title("Histogram of Spleen Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.show()

def plotHounsfildUnitSpleen():
    spleen_avg = np.average(spleen_values)
    spleen_std = np.std(spleen_values)
    n, bins, patches = plt.hist(spleen_values, 60, density=1)
    pdf_spleen = norm.pdf(bins, spleen_avg, spleen_std)
    plt.plot(bins, pdf_spleen)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title('Spleen values in CT scan')
    plt.show()
plotHounsfildUnitSpleen()


#Exercise 4
# Load the bone ROI and compute the Hounsfield units in the bone and spleen
bone_roi = io.imread(in_dir + 'BoneROI.png')
# convert to boolean image
bone_mask = bone_roi > 0
bone_values = img[bone_mask]


def plotHounsFieldBoneSpleen():
    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    mu_spleen = np.average(spleen_values)
    std_spleen = np.std(spleen_values)
    mu_bone = np.average(bone_values)
    std_bone = np.std(bone_values)
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
    plt.plot(hu_range, pdf_bone, 'g', label="bone")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()
plotHounsFieldBoneSpleen()


#Spleen liver and kidneys are hard to distinguish
#Exercise 5
def plotHounsFieldMultipleOrgans():
    # Load ROIs for fat, kidneys, and liver
    fat_roi = io.imread(in_dir + 'FatROI.png')
    kidneys_roi = io.imread(in_dir + 'KidneyROI.png')
    liver_roi = io.imread(in_dir + 'LiverROI.png')

    # Convert to boolean masks
    fat_mask = fat_roi > 0
    kidneys_mask = kidneys_roi > 0
    liver_mask = liver_roi > 0

    # Extract pixel values
    fat_values = img[fat_mask]
    kidneys_values = img[kidneys_mask]
    liver_values = img[liver_mask]

    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)

    # Compute Gaussian parameters and PDFs
    organs = {
        "spleen": (np.average(spleen_values), np.std(spleen_values)),
        "bone": (np.average(bone_values), np.std(bone_values)),
        "fat": (np.average(fat_values), np.std(fat_values)),
        "kidneys": (np.average(kidneys_values), np.std(kidneys_values)),
        "liver": (np.average(liver_values), np.std(liver_values)),
    }

    for organ, (mu, std) in organs.items():
        pdf = norm.pdf(hu_range, mu, std)
        plt.plot(hu_range, pdf, label=organ)

    plt.title("Fitted Gaussians for Multiple Organs")
    plt.xlabel("Hounsfield Unit")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

plotHounsFieldMultipleOrgans()

# Exercise 6
# Define the classes for classification
def define_classes():
    # Combine some classes into broader categories
    classes = {
        "background": (-1000, -200),  # Air and other low-density regions
        "fat": (-200, 0),            # Fat tissue
        "soft_tissue": (0, 300),     # Organs like spleen, liver, kidneys
        "bone": (300, 2000)          # Bone and other high-density regions
    }
    return classes

# Example usage
classes = define_classes()
for class_name, (lower, upper) in classes.items():
    print(f"Class '{class_name}': Hounsfield range {lower} to {upper}")

# Exercise 7/8
def visulaizeAllElements():
    # Compute the class ranges and create binary masks for each class
    t_background = -200
    t_fat_soft = 0
    t_soft_bone = 300

    # Create binary masks for each class
    background_img = img <= t_background
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_tissue_img = (img > t_fat_soft) & (img <= t_soft_bone)
    bone_img = img > t_soft_bone

    # Visualize the binary masks
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(background_img, cmap='gray')
    axes[0].set_title('Background')
    axes[0].axis('off')

    axes[1].imshow(fat_img, cmap='gray')
    axes[1].set_title('Fat')
    axes[1].axis('off')

    axes[2].imshow(soft_tissue_img, cmap='gray')
    axes[2].set_title('Soft Tissue')
    axes[2].axis('off')

    axes[3].imshow(bone_img, cmap='gray')
    axes[3].set_title('Bone')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

# Exercise 9
def colorVisuliazation():
    # Update the colors for better visualization
    # Visualize the classification result
    classified_img = np.zeros_like(img, dtype=np.uint8)
    
    # Compute the class ranges and create binary masks for each class
    t_background = -200
    t_fat_soft = 0
    t_soft_bone = 300

    # Create binary masks for each class
    background_img = img <= t_background
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_tissue_img = (img > t_fat_soft) & (img <= t_soft_bone)
    bone_img = img > t_soft_bone

    # Assign unique values to each class
    classified_img[background_img] = 1  # Background
    classified_img[fat_img] = 2         # Fat
    classified_img[soft_tissue_img] = 3 # Soft Tissue
    classified_img[bone_img] = 4        # Bone

    # Map the classified image to colors for better visualization
    colored_classified_img = label2rgb(classified_img, bg_label=0, colors=['black', 'yellow', 'green', 'blue'])

    # Show the original anatomical image and the classified result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap='gray', vmin=-200, vmax=500)
    axes[0].set_title('Original Anatomical Image')
    axes[0].axis('off')

    axes[1].imshow(colored_classified_img)
    axes[1].set_title('Classified Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
colorVisuliazation()

#Exercise 10
def plot_gaussian_intersections():
    # Define the Hounsfield unit range
    hu_range = np.linspace(-8000, 8000, 4000)

    # Compute Gaussian parameters for each class
    organs = {
        "background": (-1000, 200),  # Background
        "fat": (-200, 0),           # Fat
        "soft_tissue": (0, 300),    # Soft Tissue
        "bone": (300, 2000)         # Bone
    }

    # Fit Gaussians and plot
    for organ, (mu, std) in organs.items():
        pdf = norm.pdf(hu_range, mu, std)
        plt.plot(hu_range, pdf, label=organ)

    plt.title("Fitted Gaussians for Training Values")
    plt.xlabel("Hounsfield Unit")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Call the function to plot
plot_gaussian_intersections()

# Exercise 11
# Yes it became useless
def visualize_classification_results():
    # Define the class ranges
    t_background = -550
    t_fat_soft = 600
    t_soft_bone = 6000

    # Create binary masks for each class
    background_img = img <= t_background
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_tissue_img = (img > t_fat_soft) & (img <= t_soft_bone)
    bone_img = img > t_soft_bone

    # Assign unique values to each class
    classified_img = np.zeros_like(img, dtype=np.uint8)
    classified_img[background_img] = 1  # Background
    classified_img[fat_img] = 2         # Fat
    classified_img[soft_tissue_img] = 3 # Soft Tissue
    classified_img[bone_img] = 4        # Bone

    # Map the classified image to colors for better visualization
    colored_classified_img = label2rgb(classified_img, bg_label=0, colors=['black', 'yellow', 'green', 'blue'])

    # Show the original anatomical image and the classified result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap='gray', vmin=-200, vmax=500)
    axes[0].set_title('Original Anatomical Image')
    axes[0].axis('off')

    axes[1].imshow(colored_classified_img)
    axes[1].set_title('Classified Image (Exercise 11)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to visualize the classification results
visualize_classification_results()

# Exercise 12
def find_optimal_class_ranges():
    # Define the Gaussian parameters for each class
    classes = {
        "fat": (-200, 50),          # Mean and std for fat
        "soft_tissue": (50, 100),   # Mean and std for soft tissue
        "bone": (300, 200)          # Mean and std for bone
    }

    # Define the Hounsfield unit range
    hu_range = np.arange(-200, 1000, 1)

    # Create a dictionary to store the most probable class for each HU value
    lookup_table = {}

    for hu in hu_range:
        max_prob = -1
        best_class = None
        for class_name, (mu, std) in classes.items():
            prob = norm.pdf(hu, mu, std)
            if prob > max_prob:
                max_prob = prob
                best_class = class_name
        lookup_table[hu] = best_class

    # Find the HU values where the class changes
    boundaries = []
    prev_class = None
    for hu, current_class in lookup_table.items():
        if current_class != prev_class:
            boundaries.append((prev_class, current_class, hu))
            prev_class = current_class

    # Print the boundaries
    for prev_class, current_class, hu in boundaries[1:]:
        print(f"Boundary between {prev_class} and {current_class} at HU = {hu}")

# Call the function to find and print the optimal class ranges
find_optimal_class_ranges()


# Bottom 27 top 73

# Exercise 13
def automaticSpleenSegmentationV1():
    t_1 = 27
    t_2 = 73
    spleen_estimate = (img > t_1) & (img < t_2)
    
    footprint = disk(1)
    closed = binary_closing(spleen_estimate, footprint)

    footprint = disk(7)
    opened = binary_opening(closed, footprint)
    spleen_label_colour = color.label2rgb(opened)
    label_img = measure.label(opened)
    min_area = 4000
    max_area = 4700

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    region_props = measure.regionprops(label_img)
    for region in region_props:
        # Extract features for each region
        area = region.area
        perimeter = region.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Define feature value limits for the spleen
        if area < min_area or area > max_area or circularity < 0.5 or circularity > 1.2:
            # Set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img, i_area, 'Found spleen based on area')

automaticSpleenSegmentationV1()

def spleen_finder(img):
    # Define thresholds for spleen Hounsfield units
    t_1 = 27
    t_2 = 73

    # Create an initial binary mask for the spleen
    spleen_estimate = (img > t_1) & (img < t_2)

    # Apply morphological operations to refine the mask
    footprint_close = disk(1)
    closed = binary_closing(spleen_estimate, footprint_close)

    footprint_open = disk(7)
    opened = binary_opening(closed, footprint_open)

    # Label connected regions
    label_img = measure.label(opened)

    # Filter regions based on area and circularity
    min_area = 4000
    max_area = 4700
    label_img_filter = label_img.copy()
    region_props = measure.regionprops(label_img)

    for region in region_props:
        area = region.area
        perimeter = region.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Remove regions that do not match spleen characteristics
        if area < min_area or area > max_area or circularity < 0.5 or circularity > 1.2:
            for coords in region.coords:
                label_img_filter[coords[0], coords[1]] = 0

    # Return the final binary mask
    spleen_binary = label_img_filter > 0
    return spleen_binary

def calculate_dice_score(ground_truth, prediction):
    """
    Calculate the DICE score between the ground truth and the predicted segmentation.

    Parameters:
    - ground_truth: Binary mask of the ground truth segmentation.
    - prediction: Binary mask of the predicted segmentation.

    Returns:
    - dice_score: The DICE coefficient.
    """
    intersection = np.sum(ground_truth * prediction)
    total = np.sum(ground_truth) + np.sum(prediction)
    dice_score = (2 * intersection) / total if total > 0 else 0
    return dice_score

# Load the ground truth spleen mask
ground_truth_spleen = io.imread(in_dir + 'Validation3_spleen.png') > 0

# Predict the spleen segmentation
predicted_spleen = spleen_finder(img)

# Calculate the DICE score
dice_score = calculate_dice_score(ground_truth_spleen, predicted_spleen)
print(f"DICE Score for spleen segmentation: {dice_score:.4f}")

#Exercise 21
"""
Results:
0.8915
0.9099
0.9661
"""



def plotHounsfildUnitSpleen():
    spleen_avg = np.average(spleen_values)
    spleen_std = np.std(spleen_values)
    n, bins, patches = plt.hist(spleen_values, 60, density=1)
    pdf_spleen = norm.pdf(bins, spleen_avg, spleen_std)
    plt.plot(bins, pdf_spleen)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title('Spleen values in CT scan')
    plt.show()
plotHounsfildUnitSpleen()