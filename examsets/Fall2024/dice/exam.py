import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import norm
from scipy.optimize import brentq


# Load image
image = imageio.v2.imread('CubesG.png')

# ROI filenames
roi_files = {
    'A': 'A_Cubes.txt',
    'B': 'B_Cubes.txt',
    'C': 'C_Cubes.txt',
    'D': 'D_Cubes.txt',
    'E': 'E_Cubes.txt'
}

# Read ROI data
roi_data = {label: np.loadtxt(fname) for label, fname in roi_files.items()}

# Selected combinations to validate
combinations = [
    ('C', 'D', 'E'),
    ('B', 'D', 'E'),
    ('A', 'B', 'C'),
    ('B', 'C', 'D'),
    ('A', 'D', 'E')
]

# Minimum distance classification
def segment_image(image, roi_combination):
    means = np.array([np.mean(roi_data[label]) for label in roi_combination])
    segmented = np.zeros_like(image)
    foreground_mask = image > 0
    diffs = np.abs(image[foreground_mask, None] - means)
    labels = np.argmin(diffs, axis=1) + 1  # Labels: 1, 2, 3
    segmented[foreground_mask] = labels
    return segmented

# Load pixel values from text files
roi_d = np.loadtxt('D_Cubes.txt')
roi_e = np.loadtxt('E_Cubes.txt')

# Estimate mean and standard deviation
mu_d, sigma_d = np.mean(roi_d), np.std(roi_d)
mu_e, sigma_e = np.mean(roi_e), np.std(roi_e)

# Define function for difference in PDFs
def diff_pdfs(x):
    return norm.pdf(x, mu_d, sigma_d) - norm.pdf(x, mu_e, sigma_e)

# Use brentq to find intersection point
# Assumes that mu_d < mu_e (adjust range otherwise)
intersection = brentq(diff_pdfs, mu_d, mu_e)

# Plot the distributions
x_vals = np.linspace(0, 255, 1000)
pdf_d = norm.pdf(x_vals, mu_d, sigma_d)
pdf_e = norm.pdf(x_vals, mu_e, sigma_e)

plt.plot(x_vals, pdf_d, label='ROI D')
plt.plot(x_vals, pdf_e, label='ROI E')
plt.axvline(intersection, color='r', linestyle='--', label=f'Threshold: {intersection:.2f}')
plt.legend()
plt.xlabel('Pixel Value')
plt.ylabel('Probability Density')
plt.title('Gaussian PDFs and Optimal Threshold')
plt.grid(True)
plt.show()

print(f"Optimal threshold (intersection of PDFs): {intersection:.2f}")


# Segment and visualize each combination
for combo in combinations:
    seg = segment_image(image, combo)
    plt.imshow(seg, cmap='nipy_spectral')
    plt.title(f'Segmentation using ROIs: {combo}')
    plt.axis('off')
    plt.show()
