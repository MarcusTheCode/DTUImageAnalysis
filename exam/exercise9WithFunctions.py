import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from combined import (
    show_orthogonal_views,
    overlay_orthogonal_views,
    rotation_matrix,
    create_affine_matrix,
    resample_image,
    save_image,
    homogeneous_matrix_from_transform,
    composite2affine,
    combine_transforms_to_affine
)

dir_in = './data/3d/'

def run_exercise1():
    vol_sitk = sitk.ReadImage(dir_in + 'ImgT1.nii')
    show_orthogonal_views(vol_sitk, title='T1.nii')

def run_exercise2():
    angle = 25
    affine_matrix = create_affine_matrix(angle)
    print("4x4 Affine matrix with 25 degrees pitch:\n", affine_matrix)

def run_exercise3():
    vol_sitk = sitk.ReadImage(dir_in + 'ImgT1.nii')
    angle = 25  # degrees

    # Compute the center in physical coordinates
    center_index = np.array(vol_sitk.GetSize()) / 2 - 0.5
    center_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(center_index)

    # Get 3x3 rotation and expand it to 4x4
    rot_3x3 = rotation_matrix(angle, 0, 0)[:3, :3]
    affine_4x4 = np.eye(4)
    affine_4x4[:3, :3] = rot_3x3
    affine_4x4[:3, 3] = center_world - rot_3x3 @ center_world  # rotate around center

    # Apply transformation
    rotated_img = resample_image(vol_sitk, affine_4x4)
    save_image(rotated_img, dir_in + 'ImgT1_A.nii')

def run_exercise4():
    vol_sitk = sitk.ReadImage(dir_in + 'ImgT1.nii')
    imgT1_A = sitk.ReadImage(dir_in + 'ImgT1_A.nii')
    show_orthogonal_views(imgT1_A, title='T1_A.nii new')
    overlay_orthogonal_views(vol_sitk, imgT1_A, title='ImgT1 (red) vs. ImgT1_A (green) new')

def run_exercise5():
    fixed = sitk.ReadImage(dir_in + 'ImgT1.nii', sitk.sitkFloat32)
    moving = sitk.ReadImage(dir_in + 'ImgT1_A.nii', sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(fixed.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    save_image(resampled, dir_in + 'ImgT1_B.nii')

    affine = composite2affine(final_transform)
    affine_matrix = homogeneous_matrix_from_transform(affine)
    np.savetxt(dir_in + 'A1.txt', affine_matrix)
    show_orthogonal_views(resampled, title='Registered Result (ImgT1_B)')

    matrix_applied = create_affine_matrix(25)
    matrix_estimated = homogeneous_matrix_from_transform(final_transform.GetNthTransform(0))
    print("Applied Affine Matrix:\n", matrix_applied)
    print("Estimated Affine Matrix:\n", matrix_estimated)

def run_exercise6():
    img = sitk.ReadImage(dir_in + 'ImgT1_B.nii')
    show_orthogonal_views(img, title='Ortho view of ImgT1_B.nii')
    matrix = np.loadtxt(dir_in + 'A1.txt')
    print("Affine transformation matrix (A1.txt):\n", matrix)

def run_exercise7():
    fixed = sitk.ReadImage(dir_in + 'ImgT1.nii', sitk.sitkFloat32)
    moving = sitk.ReadImage(dir_in + 'ImgT1_A.nii', sitk.sitkFloat32)
    center_index = np.array(fixed.GetSize()) / 2.0
    center_physical = fixed.TransformContinuousIndexToPhysicalPoint(center_index)
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.5)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)
    R.SetShrinkFactorsPerLevel(shrinkFactors=[20])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(transform, inPlace=False)

    final_transform = R.Execute(fixed, moving)
    resampled = sitk.Resample(moving, fixed, final_transform)
    save_image(resampled, dir_in + 'ImgT1_B_centered.nii')
    show_orthogonal_views(resampled, title='Centered Transform Result')

def run_exercise8():
    base_img = sitk.ReadImage(dir_in + 'ImgT1.nii')
    center_index = np.array(base_img.GetSize()) / 2 - 0.5
    center_world = base_img.TransformContinuousIndexToPhysicalPoint(center_index)

    angles = [0, 60, 120, 180, 240]
    for angle in angles:
        rot_3x3 = rotation_matrix(0, 0, -angle)[:3, :3]
        affine_4x4 = np.eye(4)
        affine_4x4[:3, :3] = rot_3x3
        affine_4x4[:3, 3] = center_world - rot_3x3 @ center_world

        # Apply transformation and save
        rotated_img = resample_image(base_img, affine_4x4)
        filename = f'ImgT1_{angle}.nii'
        save_image(rotated_img, dir_in + filename)

        # Display image
        show_orthogonal_views(rotated_img, title=f'T1_{angle}.nii')

def run_exercise9():
    angles = [0, 60, 180, 240]
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1_120.nii')

    for angle in angles:
        moving_image = sitk.ReadImage(dir_in + f'ImgT1_{angle}.nii')

        # Set the registration method
        R = sitk.ImageRegistrationMethod()

        # Set pyramid strategy
        R.SetShrinkFactorsPerLevel(shrinkFactors=[20])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Interpolator
        R.SetInterpolator(sitk.sitkLinear)

        # Similarity metric
        R.SetMetricAsMeanSquares()

        # Sampling strategy
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.50)

        # Optimizer
        R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

        # Centered initializer
        initTransform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetInitialTransform(initTransform, inPlace=False)

        try:
            tform_reg = R.Execute(fixed_image, moving_image)
            tform = tform_reg.GetNthTransform(0)
            resliced_image = resample_image(moving_image, homogeneous_matrix_from_transform(tform))

            # Save and show
            sitk.WriteTransform(tform_reg, dir_in + f'Ex9_{angle}.tfm')
            save_image(resliced_image, dir_in + f'ImgT1_resliced_{angle}.nii')
            show_orthogonal_views(resliced_image, title=f'Resliced T1_{angle}.nii')

            matrix_estimated = homogeneous_matrix_from_transform(tform)
            print(f"Estimated Transformation Matrix for {angle} degrees:")
            print(matrix_estimated)

        except Exception as e:
            print(f"Registration failed for ImgT1_{angle}: {e}")

def run_exercise10():
    fixed = sitk.ReadImage(dir_in + 'ImgT1_240.nii', sitk.sitkFloat32)
    moving = sitk.ReadImage(dir_in + 'ImgT1.nii', sitk.sitkFloat32)
    tform_60 = sitk.ReadTransform(dir_in + 'Ex9_60.tfm')
    tform_180 = sitk.ReadTransform(dir_in + 'Ex9_180.tfm')
    tform_240 = sitk.ReadTransform(dir_in + 'Ex9_240.tfm')
    tform_0 = sitk.ReadTransform(dir_in + 'Ex9_0.tfm')
    center = moving.TransformContinuousIndexToPhysicalPoint(np.array(moving.GetSize()) / 2.0)

    composite = sitk.CompositeTransform(3)
    for t in [tform_240, tform_180, tform_60, tform_0]:
        composite.AddTransform(t.GetNthTransform(0))

    affine_composite = composite2affine(composite, center)
    resampled = sitk.Resample(moving, fixed, affine_composite, sitk.sitkLinear, 0.0, moving.GetPixelID())
    save_image(resampled, dir_in + 'ImgT1_combined_final.nii')
    show_orthogonal_views(resampled, title='Ortho view of ImgT1_combined_final.nii')
    print("Combined affine matrix:\n", homogeneous_matrix_from_transform(affine_composite))

def run_exercise11():
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_240.nii')

    noise_std_dev = 200
    step_lengths = [10, 50, 150, 200]

    for step_length in step_lengths:
        print(f"Testing with step length: {step_length}")

        moving_image_noisy = sitk.AdditiveGaussianNoise(moving_image, mean=0, standardDeviation=noise_std_dev)
        show_orthogonal_views(moving_image_noisy, title=f'Moving Image with Noise (Step Length: {step_length})')

        R = sitk.ImageRegistrationMethod()
        R.SetShrinkFactorsPerLevel([2, 2, 2])
        R.SetSmoothingSigmasPerLevel([3, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInterpolator(sitk.sitkLinear)
        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.50)
        R.SetOptimizerAsPowell(stepLength=step_length, numberOfIterations=50)

        initTransform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image_noisy, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetInitialTransform(initTransform, inPlace=False)

        try:
            tform_reg = R.Execute(fixed_image, moving_image_noisy)
            tform = tform_reg.GetNthTransform(0)
            affine_matrix = homogeneous_matrix_from_transform(tform)

            resliced_image = resample_image(moving_image_noisy, affine_matrix)
            save_image(resliced_image, dir_in + f'ImgT1_resliced_noise_step_{step_length}.nii')

            show_orthogonal_views(resliced_image, title=f'Resliced Image (Step Length: {step_length})')
            print(f"Estimated Transformation Matrix for Step Length {step_length}:\n{affine_matrix}")
        except Exception as e:
            print(f"Registration failed for step length {step_length}: {e}")

def run_exercise12():
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_240.nii')

    noise_std_dev = 200
    step_lengths = [10, 50, 150, 200]
    sigma_levels = [[3.0, 1.0, 0.0], [5.0, 1.0, 0.0]]  # Suggested sigma pyramid levels

    for sigma in sigma_levels:
        print(f"Testing with sigma levels: {sigma}")
        for step_length in step_lengths:
            print(f"  Testing with step length: {step_length}")

            # Add Gaussian noise to the moving image
            moving_noisy = sitk.AdditiveGaussianNoise(moving_image, mean=0, standardDeviation=noise_std_dev)
            show_orthogonal_views(moving_noisy, title=f"Noisy Image σ={sigma}, Step={step_length}")

            # Set up registration method
            R = sitk.ImageRegistrationMethod()
            R.SetShrinkFactorsPerLevel([2, 2, 2])
            R.SetSmoothingSigmasPerLevel(sigma)
            R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            R.SetInterpolator(sitk.sitkLinear)
            R.SetMetricAsMeanSquares()
            R.SetMetricSamplingStrategy(R.RANDOM)
            R.SetMetricSamplingPercentage(0.5)
            R.SetOptimizerAsPowell(stepLength=step_length, numberOfIterations=50)

            # Centered rigid transform initializer
            init_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_noisy,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            R.SetInitialTransform(init_transform, inPlace=False)

            # Execute registration and apply transform
            try:
                final_transform = R.Execute(fixed_image, moving_noisy)
                affine_matrix = homogeneous_matrix_from_transform(final_transform.GetNthTransform(0))
                resliced = resample_image(moving_noisy, affine_matrix)

                # Save and show
                filename = f'ImgT1_resliced_sigma{int(sigma[0])}_step{step_length}.nii'
                save_image(resliced, dir_in + filename)
                show_orthogonal_views(resliced, title=f"Resliced σ={sigma}, Step={step_length}")

                print(f"    Affine Matrix (σ={sigma}, step={step_length}):\n{affine_matrix}")
            except Exception as e:
                print(f"    Registration failed (σ={sigma}, step={step_length}): {e}")


run_exercise9()