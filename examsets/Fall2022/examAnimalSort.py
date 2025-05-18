import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Training data
cows = np.array([26, 46, 33, 23, 35, 28, 21, 30, 38, 43])
sheep = np.array([67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100])

# Load from file if needed
# cows = np.loadtxt("cows.txt")
# sheep = np.loadtxt("sheep.txt")

# Minimum Distance Classifier
mean_cow = np.mean(cows)
mean_sheep = np.mean(sheep)
threshold_md = (mean_cow + mean_sheep) / 2

# Parametric Classification (Gaussian)
mu_cow, std_cow = np.mean(cows), np.std(cows)
mu_sheep, std_sheep = np.mean(sheep), np.std(sheep)

# Define intensity range
x = np.linspace(0, 120, 500)
pdf_cow = norm.pdf(x, mu_cow, std_cow)
pdf_sheep = norm.pdf(x, mu_sheep, std_sheep)

# Find intersection for threshold
diff = np.abs(pdf_cow - pdf_sheep)
threshold_parametric = x[np.argmin(diff)]

# Value for inspection
value_check = 38
pdf_value_cow = norm.pdf(value_check, mu_cow, std_cow)
pdf_value_sheep = norm.pdf(value_check, mu_sheep, std_sheep)

# Output results
print(f"Minimum Distance Threshold: {threshold_md:.2f}")
print(f"Parametric Threshold: {threshold_parametric:.2f}")
print(f"Gaussian value for cows at 38: {pdf_value_cow:.4f}")
print(f"Gaussian value for sheep at 38: {pdf_value_sheep:.4f}")

# Visualization
plt.plot(x, pdf_cow, label='Cow Gaussian')
plt.plot(x, pdf_sheep, label='Sheep Gaussian')
plt.axvline(threshold_md, color='green', linestyle='--', label='Min Distance Threshold')
plt.axvline(threshold_parametric, color='red', linestyle='--', label='Parametric Threshold')
plt.scatter([value_check], [pdf_value_cow], color='blue', label=f'Cow @ {value_check}')
plt.scatter([value_check], [pdf_value_sheep], color='orange', label=f'Sheep @ {value_check}')
plt.legend()
plt.title("Gaussian Models and Thresholds")
plt.xlabel("Average Intensity")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
