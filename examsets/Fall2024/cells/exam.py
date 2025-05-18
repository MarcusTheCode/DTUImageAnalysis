import numpy as np
import imageio
from skimage.color import rgb2gray
from skimage.morphology import disk, opening
from skimage.measure import label

def find_landmarks(image_path):
    # Read the image as grayscale
    image = imageio.v2.imread(image_path)
    
    # Find coordinates where pixel values are between 1 and 5
    landmarks = []
    for label in range(1, 6):
        coords = np.argwhere(image == label)
        for coord in coords:
            y, x = coord  # row=y, col=x
            landmarks.append((x, y))
    return landmarks

# Example usage:
fixed_landmarks = find_landmarks("LabelsFixedImg.png")
moving_landmarks = find_landmarks("LabelsMovingImg.png")

print("Fixed Image Landmarks:", fixed_landmarks)
print("Moving Image Landmarks:", moving_landmarks)


def compute_false_negatives(true_positives, false_positives, sensitivity):
    # Using the formula: Sensitivity = TP / (TP + FN)
    fn = (true_positives / sensitivity) - true_positives
    return round(fn)

# Example values from the problem
TP = 18
FP = 7
sensitivity = 0.82

missed_cells = compute_false_negatives(TP, FP, sensitivity)
print("Missed Cells (False Negatives):", missed_cells)

def preprocess_and_count_cells(image_path, threshold=30, selem_size=3):
    # Load image
    image = imageio.v2.imread(image_path)

    # Convert to grayscale if image is RGB
    if image.ndim == 3:
        image = rgb2gray(image)
        image = (image * 255).astype(np.uint8)  # Convert to 0-255 scale

    # Thresholding
    binary_image = image > threshold

    # Morphological opening with a disk-shaped structuring element
    selem = disk(selem_size)
    cleaned_image = opening(binary_image, selem)

    # BLOB analysis (connected component labeling)
    labeled_image = label(cleaned_image, connectivity=2)

    # Count number of unique labels (excluding background = 0)
    num_cells = labeled_image.max()
    return num_cells

# Apply to both images
cells_x = preprocess_and_count_cells("x_NisslStain_9-260.81.png")
cells_y = preprocess_and_count_cells("y_NisslStain_9-260.81.png")

print(f"Number of cells in x_NisslStain_9-260.81.png: {cells_x}")
print(f"Number of cells in y_NisslStain_9-260.81.png: {cells_y}")

def average_landmark(landmarks):
    return np.mean(landmarks, axis=0)

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Step 1: Compute average positions
avg_fixed = average_landmark(fixed_landmarks)
avg_moving = average_landmark(moving_landmarks)

# Step 2: Compute Euclidean distance between the average points
distance = euclidean_distance(avg_fixed, avg_moving)
print("Average landmark distance:", distance)