import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage import io, color
from skimage.color import label2rgb
import pydicom
from combined import *

# Set data folder
DATA_FOLDER = "data/pixelClass/"

# Exercise 1: Visualize CT scan with proper HU contrast
def exercise_1():
    ct = pydicom.read_file(os.path.join(DATA_FOLDER, "Training.dcm"))
    img = ct.pixel_array
    plt.imshow(img, cmap="gray", vmin=0, vmax=150)
    plt.title("CT scan - spleen contrast")
    plt.axis("off")
    plt.show()
    return img

# Exercise 2: Compute mean and std of spleen HU values
def exercise_2(img):
    spleen_mask = io.imread(os.path.join(DATA_FOLDER, "SpleenROI.png")) > 0
    spleen_values = img[spleen_mask]
    mu, std = np.mean(spleen_values), np.std(spleen_values)
    print(f"Spleen mean HU: {mu:.2f}, std: {std:.2f}")
    return spleen_values

# Exercise 3: Plot histogram of spleen pixel values
def exercise_3(values):
    plt.hist(values, bins=50)
    plt.title("Spleen HU histogram")
    plt.xlabel("Hounsfield Units")
    plt.ylabel("Frequency")
    plt.show()

# Exercise 4: Plot histogram with Gaussian fit
def exercise_4(values, title="Spleen HU + Gaussian"):
    mu, std = np.mean(values), np.std(values)
    n, bins, _ = plt.hist(values, bins=60, density=1, alpha=0.6)
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf, 'r')
    plt.title(title)
    plt.xlabel("Hounsfield Unit")
    plt.ylabel("Probability density")
    plt.show()
    return mu, std

# Exercise 5: Compare organ Gaussian distributions
def exercise_5(img):
    organs = ["Bone", "Fat", "Kidney", "Liver", "Spleen"]
    hu_range = np.arange(-200, 1000, 1.0)
    organ_values = {}
    for name in organs:
        mask = io.imread(os.path.join(DATA_FOLDER, f"{name}ROI.png")) > 0
        values = img[mask]
        organ_values[name] = values
    plot_organ_gaussians(organ_values, hu_range)
    return organ_values

# Exercise 6: Show defined classes
def exercise_6():
    print("Defined classes:")
    for k, v in define_classes().items():
        print(f"{k}: {v}")

# Exercise 7-9: Minimum distance classifier & visualization
def exercise_7_to_9(img, organ_values):
    thresholds = {}
    means = {k: np.mean(v) for k, v in organ_values.items() if k in ["Fat", "Kidney", "Liver", "Bone"]}
    means["SoftTissue"] = np.mean([means["Kidney"], means["Liver"], np.mean(organ_values["Spleen"])])
    thresholds["background_fat"] = -200
    thresholds["fat_soft"] = (means["Fat"] + means["SoftTissue"]) / 2
    thresholds["soft_bone"] = (means["SoftTissue"] + means["Bone"]) / 2

    fat_img = (img > thresholds["background_fat"]) & (img <= thresholds["fat_soft"])
    soft_img = (img > thresholds["fat_soft"]) & (img <= thresholds["soft_bone"])
    bone_img = img > thresholds["soft_bone"]

    label_img = fat_img.astype(int) + 2 * soft_img.astype(int) + 3 * bone_img.astype(int)
    overlay = label2rgb(label_img)
    show_comparison(img, overlay, "Minimum distance classification")

# Exercise 10-12: Parametric classifier by probability
def exercise_10_to_12(organ_values):
    params = {k: (np.mean(v), np.std(v)) for k, v in organ_values.items()}
    transitions = find_class_boundaries_gaussian(params)
    print("Class boundaries (HU transitions):")
    for prev, cls, hu in transitions:
        print(f"{prev or 'Start'} → {cls} at HU = {hu:.1f}")

# Exercise 13-21: Spleen segmentation and evaluation
def exercise_13_to_21(img):
    for val_name in ["Validation1", "Validation2", "Validation3"]:
        pred = spleen_finder(img)
        gt_path = (DATA_FOLDER + f"{val_name}_spleen.png")
        gt = io.imread(gt_path)
        show_comparison(gt, pred, f"{val_name} - Ground Truth vs Prediction")
        dice = calculate_dice_score(pred, gt)
        print(f"{val_name} DICE score: {dice:.4f}")

# Run all exercises
def main():
    img = exercise_1()
    spleen_values = exercise_2(img)
    exercise_3(spleen_values)
    mu, std = exercise_4(spleen_values)
    organ_values = exercise_5(img)
    exercise_6()
    exercise_7_to_9(img, organ_values)
    exercise_10_to_12(organ_values)
    exercise_13_to_21(img)

if __name__ == "__main__":
    main()
