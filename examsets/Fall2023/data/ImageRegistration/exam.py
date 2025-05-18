import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

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

def quick_overlay_slices(sitk_img_red, sitk_img_green, origin=None, title=None):
    vol_r = sitk.GetArrayFromImage(sitk_img_red)
    vol_g = sitk.GetArrayFromImage(sitk_img_green)
    vol_r[vol_r < 0] = 0
    vol_g[vol_g < 0] = 0
    R = img_as_ubyte(vol_r / np.max(vol_r))
    G = img_as_ubyte(vol_g / np.max(vol_g))
    if origin is None:
        origin = np.array(R.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.stack([R[origin[0], ::-1, ::-1], G[origin[0], ::-1, ::-1], np.zeros_like(R[origin[0]])], axis=-1))
    axes[0].set_title('Axial')
    axes[1].imshow(np.stack([R[::-1, origin[1], ::-1], G[::-1, origin[1], ::-1], np.zeros_like(R[:, origin[1]])], axis=-1))
    axes[1].set_title('Coronal')
    axes[2].imshow(np.stack([R[::-1, ::-1, origin[2]], G[::-1, ::-1, origin[2]], np.zeros_like(R[:, :, origin[2]])], axis=-1))
    axes[2].set_title('Sagittal')
    [ax.set_axis_off() for ax in axes]
    if title:
        fig.suptitle(title)
    plt.show()

# Load images
folder_in = ''
fixedImage = sitk.ReadImage(folder_in + 'ImgT1_v1.nii.gz', sitk.sitkFloat32)
movingImage = sitk.ReadImage(folder_in + 'ImgT1_v2.nii.gz', sitk.sitkFloat32)

# Step 1: Apply -20 degrees roll to ImgT1_v2 using default center of rotation
manual_transform = sitk.Euler3DTransform()
manual_transform.SetRotation(0.0, np.deg2rad(-20), 0.0)  # Rx=0, Ry=-20°, Rz=0

# Set default center (image center in physical space)
center = movingImage.TransformIndexToPhysicalPoint([sz // 2 for sz in movingImage.GetSize()])
manual_transform.SetCenter(center)

# Apply the transform
moving_manual = sitk.Resample(movingImage, fixedImage, manual_transform, sitk.sitkLinear, 0.0)

# Step 2: Create brain mask from ImgT1_v1 (intensity > 50)
brain_mask = sitk.Cast(fixedImage > 50, sitk.sitkFloat32)

# Step 3: Compute Mean Squared Error (MSE) within mask
fixed_array = sitk.GetArrayFromImage(fixedImage)
moved_array = sitk.GetArrayFromImage(moving_manual)
mask_array = sitk.GetArrayFromImage(brain_mask)

mse = np.sum(((fixed_array - moved_array) ** 2) * mask_array) / np.sum(mask_array)

print("Manual registration MSE (within brain mask):", round(mse, 2))

# Optional: Visualize
quick_overlay_slices(fixedImage, moving_manual, title='Manual Transform Overlay')
