import numpy as np
import cv2
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load images
zebra_img = cv2.imread("Zebra.png", cv2.IMREAD_GRAYSCALE)
white_mask = cv2.imread("Zebra_whiteStripes.png", cv2.IMREAD_GRAYSCALE) // 255
black_mask = cv2.imread("Zebra_blackStripes.png", cv2.IMREAD_GRAYSCALE) // 255
analysis_mask = cv2.imread("zebra_MASK.png", cv2.IMREAD_GRAYSCALE) // 255

# Extract training pixel values
white_pixels = zebra_img[white_mask == 1]
black_pixels = zebra_img[black_mask == 1]

# Estimate Gaussian parameters
mu_white, sigma_white = np.mean(white_pixels), np.std(white_pixels)
mu_black, sigma_black = np.mean(black_pixels), np.std(black_pixels)

# Generate intensity range for PDF
x_vals = np.linspace(0, 255, 1000)
pdf_white = norm.pdf(x_vals, mu_white, sigma_white)
pdf_black = norm.pdf(x_vals, mu_black, sigma_black)

# Find optimal threshold where the PDFs intersect
diff = np.abs(pdf_white - pdf_black)
optimal_threshold = x_vals[np.argmin(diff)]

# Classify pixels inside the analysis mask
masked_pixels = zebra_img[analysis_mask == 1]
classified_as_white = masked_pixels >= optimal_threshold
num_white_pixels = np.sum(classified_as_white)

# Output
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"Number of pixels classified as white stripe: {num_white_pixels}")

# Optional: plot the distributions
plt.plot(x_vals, pdf_white, label='White Stripes')
plt.plot(x_vals, pdf_black, label='Black Stripes')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Threshold = {optimal_threshold:.2f}')
plt.legend()
plt.title('PDFs of Stripe Classes and Threshold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()


# Estimate Gaussian parameters
mu_white, sigma_white = np.mean(white_pixels), np.std(white_pixels)
mu_black, sigma_black = np.mean(black_pixels), np.std(black_pixels)

# Output the Gaussian parameters for white stripes
print(f"White Stripe Gaussian Parameters:\nMean: {mu_white:.2f}, Standard Deviation: {sigma_white:.2f}")
