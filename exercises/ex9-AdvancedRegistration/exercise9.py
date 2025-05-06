import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
from skimage.util import img_as_ubyte


def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    plt.show()

def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.show()


def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.
    
    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def rotation_matrix(pitch, roll, yaw):
    """
    Create a 3D rotation matrix from pitch, roll, and yaw angles (in degrees).

    Parameters
    ----------
    pitch : float
        Rotation around the x-axis in degrees.
    roll : float
        Rotation around the y-axis in degrees.
    yaw : float
        Rotation around the z-axis in degrees.

    Returns
    -------
    np.ndarray
        A 4x4 affine transformation matrix representing the rotation.
    """
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx  # Combined rotation matrix

    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R

    return affine_matrix


dir_in = 'data/'
vol_sitk = sitk.ReadImage(dir_in + 'ImgT1.nii')

def exercise1():
    # Display the volume
    imshow_orthogonal_view(vol_sitk, title='T1.nii')

def exercise2():
    # Create a 4x4 affine matrix with a pitch of 25 degrees
    affine_matrix = rotation_matrix(25, 0, 0)
    print("Affine Matrix with 25-degree pitch:")
    print(affine_matrix)

def exercise3():
    # Define the roll rotation in radians
    angle = 25  # degrees
    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)

    centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5 # Image Coordinate System
    centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
    rot_matrix = rotation_matrix(angle, 0, 0)[:3, :3] # Ensure the rotation matrix is in degrees

    transform.SetCenter(centre_world) # Set the rotation centre
    transform.SetMatrix(rot_matrix.T.flatten())

    # Apply the transformation to the image
    ImgT1_A = sitk.Resample(vol_sitk, transform)

    # Save the rotated image
    sitk.WriteImage(ImgT1_A, dir_in + 'ImgT1_A.nii')
#exercise3()

ImgT1_A = sitk.ReadImage(dir_in + 'ImgT1_A.nii')

def exercise4():
    imshow_orthogonal_view(ImgT1_A, title='T1_A.nii')
    overlay_slices(vol_sitk, ImgT1_A, title = 'ImgT1 (red) vs. ImgT1_A (green)')
exercise4()

def homogeneous_matrix_from_transform(transform):
    """Convert a SimpleITK transform to a homogeneous matrix."""
    matrix = np.zeros((4, 4))
    if isinstance(transform, sitk.CompositeTransform):
        transform = transform.GetNthTransform(0)  # Extract the first transform if it's a composite
    matrix[:3, :3] = np.reshape(np.array(transform.GetMatrix()), (3, 3))
    matrix[:3, 3] = transform.GetTranslation()
    matrix[3, 3] = 1
    return matrix

def exercise5():

    dir_in = 'data/'
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_A.nii')

    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid scheule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors = [2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.50)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid 
    initTransform = sitk.Euler3DTransform()
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track to the optimization process 
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value
    R.AddCommand(sitk.sitkStartEvent, start_plot) # Plot the similarity metric values across iterations
    R.AddCommand(sitk.sitkEndEvent, end_plot)
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    #R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.01, numberOfIterations=500)

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixed_image, moving_image)

    # Apply the estimated transformation to the moving image
    ImgT1_B = sitk.Resample(moving_image, tform_reg)

    # Exercise 6
    # Display the optimal affine matrix found
    estimated_tform = tform_reg.GetNthTransform(0)
    matrix_estimated = homogeneous_matrix_from_transform(estimated_tform)

    print("Estimated Transformation Matrix:")
    print(matrix_estimated)
    ##### End Exercise 6 #####

    # Save 
    sitk.WriteImage(ImgT1_B, dir_in + 'ImgT1_B.nii')
    # Display the result
    overlay_slices(fixed_image, ImgT1_B, title = 'ImgT1 (red) vs. ImgT1_B (green)')
#exercise5()

def exercise7():
    dir_in = 'data/'
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_A.nii')

    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid schedule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.50)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid with the center of the fixed image
    initTransform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track of the optimization process
    R.AddCommand(sitk.sitkStartEvent, start_plot)
    R.AddCommand(sitk.sitkEndEvent, end_plot)
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixed_image, moving_image)

    # Apply the estimated transformation to the moving image
    ImgT1_C = sitk.Resample(moving_image, tform_reg)

    # Display the optimal affine matrix found
    estimated_tform = tform_reg.GetNthTransform(0)
    matrix_estimated = homogeneous_matrix_from_transform(estimated_tform)

    print("Estimated Transformation Matrix with Centered Initialization:")
    print(matrix_estimated)

    # Save
    sitk.WriteImage(ImgT1_C, dir_in + 'ImgT1_C.nii')
    # Display the result
    overlay_slices(fixed_image, ImgT1_C, title='ImgT1 (red) vs. ImgT1_C (green)')

#exercise7()

def exercise8():
    angles = [0, 60, 120, 180, 240]  # Rotation angles in degrees
    for angle in angles:
        # Create the Affine transform and set the rotation
        transform = sitk.AffineTransform(3)

        centre_image = np.array(vol_sitk.GetSize()) / 2 - 0.5  # Image Coordinate System
        centre_world = vol_sitk.TransformContinuousIndexToPhysicalPoint(centre_image)  # World Coordinate System
        rot_matrix = rotation_matrix(0, 0, angle)[:3, :3]  # Rotation around the z-axis

        transform.SetCenter(centre_world)  # Set the rotation centre
        transform.SetMatrix(rot_matrix.T.flatten())

        # Apply the transformation to the image
        rotated_image = sitk.Resample(vol_sitk, transform)

        # Save the rotated image
        output_filename = dir_in + f'ImgT1_{angle}.nii'
        sitk.WriteImage(rotated_image, output_filename)

        # Display the rotated image in orthogonal view
        imshow_orthogonal_view(rotated_image, title=f'T1_{angle}.nii')

#exercise8()

def exercise9():
    angles = [0, 60, 180, 240]  # Rotation angles in degrees
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1_120.nii')

    for angle in angles:
        moving_image = sitk.ReadImage(dir_in + f'ImgT1_{angle}.nii')

        # Set the registration
        R = sitk.ImageRegistrationMethod()

        # Set a one-level pyramid schedule
        R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set the interpolator
        R.SetInterpolator(sitk.sitkLinear)

        # Set the similarity metric
        R.SetMetricAsMeanSquares()

        # Set the sampling strategy
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.50)

        # Set the optimizer
        R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

        # Initialize the transformation type to rigid
        initTransform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetInitialTransform(initTransform, inPlace=False)

        # Estimate the registration transformation
        tform_reg = R.Execute(fixed_image, moving_image)

        # Apply the estimated transformation to the moving image
        resliced_image = sitk.Resample(moving_image, tform_reg)

        # Save the transformation
        transform_filename = dir_in + f'Ex9_{angle}.tfm'
        sitk.WriteTransform(tform_reg, transform_filename)

        # Save the resliced image
        resliced_filename = dir_in + f'ImgT1_resliced_{angle}.nii'
        sitk.WriteImage(resliced_image, resliced_filename)

        # Display the resliced image in orthogonal view
        imshow_orthogonal_view(resliced_image, title=f'Resliced T1_{angle}.nii')

        # Print the estimated transformation matrix
        estimated_tform = tform_reg.GetNthTransform(0)
        matrix_estimated = homogeneous_matrix_from_transform(estimated_tform)
        print(f"Estimated Transformation Matrix for {angle} degrees:")
        print(matrix_estimated)

#exercise9()

def compare_original_and_resliced_angles(original_angles, fixed_image_path, dir_in):
    """
    Compare the original angles with the resliced images by calculating the difference
    between the original and estimated transformation matrices.

    Parameters
    ----------
    original_angles : list
        List of original rotation angles in degrees.
    fixed_image_path : str
        Path to the fixed image used for registration.
    dir_in : str
        Directory containing the resliced images and transformations.
    """
    fixed_image = sitk.ReadImage(fixed_image_path)

    for angle in original_angles:
        # Load the resliced image and transformation
        resliced_image_path = dir_in + f'ImgT1_resliced_{angle}.nii'
        transform_path = dir_in + f'Ex9_{angle}.tfm'

        resliced_image = sitk.ReadImage(resliced_image_path)
        estimated_transform = sitk.ReadTransform(transform_path)

        # Compute the original transformation matrix
        original_transform = sitk.AffineTransform(3)
        centre_image = np.array(fixed_image.GetSize()) / 2 - 0.5
        centre_world = fixed_image.TransformContinuousIndexToPhysicalPoint(centre_image)
        rot_matrix = rotation_matrix(0, 0, angle)[:3, :3]

        original_transform.SetCenter(centre_world)
        original_transform.SetMatrix(rot_matrix.T.flatten())

        original_matrix = homogeneous_matrix_from_transform(original_transform)
        estimated_matrix = homogeneous_matrix_from_transform(estimated_transform)

        # Calculate the difference between the matrices
        matrix_difference = np.abs(original_matrix - estimated_matrix)

        print(f"Angle: {angle} degrees")
        print("Original Transformation Matrix:")
        print(original_matrix)
        print("Estimated Transformation Matrix:")
        print(estimated_matrix)
        print("Difference Matrix:")
        print(matrix_difference)
        print("Max Difference:", np.max(matrix_difference))
        print("-" * 50)

# Example usage
#angles_to_compare = [60, 180, 240]
#compare_original_and_resliced_angles(angles_to_compare, dir_in + 'ImgT1_120.nii', dir_in)

def exercise10():
    # Load the fixed and moving images
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1_240.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1.nii')

    # Load the transforms from file
    tform_60 = sitk.ReadTransform(dir_in + 'Ex9_60.tfm')
    tform_180 = sitk.ReadTransform(dir_in + 'Ex9_180.tfm')
    tform_240 = sitk.ReadTransform(dir_in + 'Ex9_240.tfm')
    tform_0 = sitk.ReadTransform(dir_in + 'Ex9_0.tfm')

    # Option A: Combine the transforms using sitk.CompositeTransform
    tform_composite = sitk.CompositeTransform(3)
    tform_composite.AddTransform(tform_240.GetNthTransform(0))
    tform_composite.AddTransform(tform_180.GetNthTransform(0))
    tform_composite.AddTransform(tform_60.GetNthTransform(0))
    tform_composite.AddTransform(tform_0.GetNthTransform(0))

    # Convert the composite transform to an affine transform
    centre_image = np.array(fixed_image.GetSize()) / 2 - 0.5
    centre_world = fixed_image.TransformContinuousIndexToPhysicalPoint(centre_image)
    affine_composite = composite2affine(tform_composite, centre_world)

    # Apply the combined affine transformation to the moving image
    resliced_image = sitk.Resample(moving_image, fixed_image, affine_composite, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Display the combined affine matrix
    combined_matrix = homogeneous_matrix_from_transform(affine_composite)
    print("Combined Affine Transformation Matrix:")
    print(combined_matrix)

    # Save the resliced image
    sitk.WriteImage(resliced_image, dir_in + 'ImgT1_combined_resliced.nii')

    # Display the resliced image in orthogonal view
    imshow_orthogonal_view(resliced_image, title='ImgT1 Combined Resliced')

# Run exercise10
#exercise10()

def exercise11():
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_240.nii')

    # Noise levels and step lengths to test
    noise_std_dev = 200
    step_lengths = [10, 50, 150, 200]

    for step_length in step_lengths:
        print(f"Testing with step length: {step_length}")

        # Add noise to the moving image
        moving_image_noisy = sitk.AdditiveGaussianNoise(moving_image, mean=0, standardDeviation=noise_std_dev)
        imshow_orthogonal_view(moving_image_noisy, title=f'Moving Image with Noise (Step Length: {step_length})')

        # Set up the registration
        R = sitk.ImageRegistrationMethod()

        # Pyramid multi-resolution strategy
        R.SetShrinkFactorsPerLevel(shrinkFactors=[2, 2, 2])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set the interpolator
        R.SetInterpolator(sitk.sitkLinear)

        # Set the similarity metric
        R.SetMetricAsMeanSquares()

        # Set the sampling strategy
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.50)

        # Set the optimizer
        R.SetOptimizerAsPowell(stepLength=step_length, numberOfIterations=50)

        # Initialize the transformation type to rigid
        initTransform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image_noisy, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetInitialTransform(initTransform, inPlace=False)

        # Estimate the registration transformation
        try:
            tform_reg = R.Execute(fixed_image, moving_image_noisy)

            # Apply the estimated transformation to the noisy moving image
            resliced_image = sitk.Resample(moving_image_noisy, tform_reg)

            # Display the resliced image
            imshow_orthogonal_view(resliced_image, title=f'Resliced Image (Step Length: {step_length})')

            # Print the estimated transformation matrix
            estimated_tform = tform_reg.GetNthTransform(0)
            matrix_estimated = homogeneous_matrix_from_transform(estimated_tform)
            print(f"Estimated Transformation Matrix for Step Length {step_length}:")
            print(matrix_estimated)
        except Exception as e:
            print(f"Registration failed for step length {step_length}: {e}")

# Run exercise11
#exercise11()

def exercise12():
    fixed_image = sitk.ReadImage(dir_in + 'ImgT1.nii')
    moving_image = sitk.ReadImage(dir_in + 'ImgT1_240.nii')

    # Noise levels and step lengths to test
    noise_std_dev = 200
    step_lengths = [10, 50, 150, 200]
    sigma_levels = [[3.0, 1.0, 0.0], [5.0, 1.0, 0.0]]  # Different sigma settings

    for sigma in sigma_levels:
        print(f"Testing with sigma levels: {sigma}")
        for step_length in step_lengths:
            print(f"  Testing with step length: {step_length}")

            # Add noise to the moving image
            moving_image_noisy = sitk.AdditiveGaussianNoise(moving_image, mean=0, standardDeviation=noise_std_dev)

            # Set up the registration
            R = sitk.ImageRegistrationMethod()

            # Pyramid multi-resolution strategy
            R.SetShrinkFactorsPerLevel(shrinkFactors=[2, 2, 2])
            R.SetSmoothingSigmasPerLevel(smoothingSigmas=sigma)
            R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Set the interpolator
            R.SetInterpolator(sitk.sitkLinear)

            # Set the similarity metric
            R.SetMetricAsMeanSquares()

            # Set the sampling strategy
            R.SetMetricSamplingStrategy(R.RANDOM)
            R.SetMetricSamplingPercentage(0.50)

            # Set the optimizer
            R.SetOptimizerAsPowell(stepLength=step_length, numberOfIterations=50)

            # Initialize the transformation type to rigid
            initTransform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image_noisy, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            R.SetInitialTransform(initTransform, inPlace=False)

            # Estimate the registration transformation
            try:
                tform_reg = R.Execute(fixed_image, moving_image_noisy)

                # Apply the estimated transformation to the noisy moving image
                resliced_image = sitk.Resample(moving_image_noisy, tform_reg)

                # Display the resliced image
                imshow_orthogonal_view(resliced_image, title=f'Resliced Image (Sigma: {sigma}, Step Length: {step_length})')

                # Print the estimated transformation matrix
                estimated_tform = tform_reg.GetNthTransform(0)
                matrix_estimated = homogeneous_matrix_from_transform(estimated_tform)
                print(f"    Estimated Transformation Matrix for Step Length {step_length}:")
                print(matrix_estimated)
            except Exception as e:
                print(f"    Registration failed for step length {step_length}: {e}")

# Run exercise12
# exercise12()