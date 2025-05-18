import numpy as np
from math import degrees
from skimage.transform import SimilarityTransform


# Reference landmarks (standing man)
reference_points = np.array([
    [1, 0],   # left leg
    [5, 0],   # right leg
    [3, 6],   # head
    [2, 4],   # left arm
    [4, 4]    # right arm
])

# Template landmarks (running man)
template_points = np.array([
    [3, 1],     # left leg
    [7, 1],     # right leg
    [4.5, 6],   # head
    [3.5, 3],   # left arm
    [5.5, 5]    # right arm
])

# --- 1. Compute the optimal translation ---
centroid_reference = np.mean(reference_points, axis=0)
centroid_template = np.mean(template_points, axis=0)
translation = centroid_reference - centroid_template

print("Optimal translation (Δx, Δy):", translation)

# --- 2. Compute the SSD before any transformation ---
differences = reference_points - template_points
ssd_before = np.sum(np.sum(differences**2, axis=1))
print("Sum-of-squared-distances (before any transformation):", ssd_before)

# --- 3. Compute the similarity transform ---
tform = SimilarityTransform()
tform.estimate(template_points, reference_points)

# Extract the absolute value of the rotation in degrees
rotation_degrees = abs(degrees(tform.rotation))
print("Absolute value of the rotation (in degrees):", rotation_degrees)
