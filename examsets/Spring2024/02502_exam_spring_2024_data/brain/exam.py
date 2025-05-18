import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.morphology import ball, closing, erosion
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

# Load utility function provided
def quick_show_orthogonal(sitk_img, origin=None, title=None):
    data = sitk.GetArrayFromImage(sitk_img)
    if origin is None:
        origin = np.array(data.shape) // 2
    data = img_as_ubyte(data / np.max(data))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')
    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')
    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')
    [ax.set_axis_off() for ax in axes]
    if title:
        fig.suptitle(title)
    plt.show()

def rotation_matrix(pitch, roll=0, yaw=0):
    pitch, roll, yaw = np.deg2rad([pitch, roll, yaw])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# Load image
template_image = sitk.ReadImage('T1_brain_template.nii.gz', sitk.sitkFloat32)

# Define rotation
R = rotation_matrix(pitch=-30, yaw=10)
transform = sitk.Euler3DTransform()
transform.SetMatrix(R.T.flatten())
transform.SetCenter(template_image.TransformContinuousIndexToPhysicalPoint(np.array(template_image.GetSize())/2.0))

# Apply transform
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(template_image)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(transform)
moving_image = resampler.Execute(template_image)

# Otsu thresholding
template_array = sitk.GetArrayFromImage(template_image)
thresh = threshold_otsu(template_array)
mask_array = template_array > thresh

# Morphological operations
mask_array = closing(mask_array, ball(5))
mask_array = erosion(mask_array, ball(3))

# Convert back to SimpleITK
mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
mask_image.CopyInformation(template_image)

# Apply mask
masked_template = sitk.Mask(template_image, mask_image)
masked_moving = sitk.Mask(moving_image, mask_image)

t = sitk.GetArrayFromImage(masked_template)
m = sitk.GetArrayFromImage(masked_moving)
mask = mask_array

# Compute NCC (normalized cross-correlation)
mean_t = np.mean(t[mask])
mean_m = np.mean(m[mask])
ncc = np.sum((t[mask] - mean_t) * (m[mask] - mean_m)) / \
      (np.sqrt(np.sum((t[mask] - mean_t)**2)) * np.sqrt(np.sum((m[mask] - mean_m)**2)))

print("Normalized Cross-Correlation Coefficient:", ncc)

quick_show_orthogonal(moving_image, title='Moving Image After Rigid Transformation')
