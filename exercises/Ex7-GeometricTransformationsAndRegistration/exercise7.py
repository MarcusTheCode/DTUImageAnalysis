import numpy as np
import math
from skimage import io, img_as_float
from skimage.transform import rotate, EuclideanTransform, SimilarityTransform, warp, swirl, matrix_transform
import cv2

import matplotlib.pyplot as plt

im_org = io.imread('./data/NusaPenida.png')
rotation_angle = 10
rotated_img = rotate(im_org, rotation_angle)

# Utility function to show two images side-by-side
def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

def exercise1_rotation(im_org, rotation_angle):
    rotated_img = rotate(im_org, rotation_angle)
    show_comparison(im_org, rotated_img, "Rotated image")
#exercise1_rotation(im_org, rotation_angle)

def exercise2_rotation_with_center(im_org, rotation_angle, rot_center):
    rotated_img_center = rotate(im_org, rotation_angle, center=rot_center)
    show_comparison(im_org, rotated_img_center, "Rotated with center (0, 0)")
#exercise2_rotation_with_center(im_org, rotation_angle, [0, 0])

"""
Reflect mirrors the content of the image onto the black background.
Wrap wraps the image around, creating a continuous effect.
Wrap places the image in a circular manner, where the edges are connected.
"""
def exercise3_rotation_with_modes(im_org, rotation_angle):
    rotated_img_reflect = rotate(im_org, rotation_angle, mode="reflect")
    rotated_img_wrap = rotate(im_org, rotation_angle, mode="wrap")
    show_comparison(im_org, rotated_img_reflect, "Reflect mode")
    show_comparison(im_org, rotated_img_wrap, "Wrap mode")
#exercise3_rotation_with_modes(im_org, rotation_angle)

"""
CVal 1 makes the background black wile cval 0.00001 makes it white.
"""
def exercise4_rotation_with_constant_fill(im_org, rotation_angle):
    rotated_img_constant = rotate(im_org, rotation_angle, resize=True, mode="constant", cval=0.00001)
    show_comparison(im_org, rotated_img_constant, "Constant mode with cval=1")
#exercise4_rotation_with_constant_fill(im_org, rotation_angle)

def exercise5_automatic_resizing(im_org, rotation_angle):
    rotated_img_resize = rotate(im_org, rotation_angle, resize=True)
    show_comparison(im_org, rotated_img_resize, "Resized image")
#exercise5_automatic_resizing(im_org, rotation_angle)
"""
Automatcially resizes the image to fit the rotated content within the box
resize false leaves content outside the box
"""

def exercise6_euclidean_transform(rotation_angle_rad, trans):
    tform = EuclideanTransform(rotation=rotation_angle_rad, translation=trans)
    print("Euclidean Transform Matrix:\n", tform.params)
    return tform
tform = exercise6_euclidean_transform(10.0 * math.pi / 180.0, [10, 20])

def exercise7_apply_euclidean_transform(im_org, tform):
    transformed_img = warp(im_org, tform)
    show_comparison(im_org, transformed_img, "Euclidean Transform")
#exercise7_apply_euclidean_transform(im_org, exercise6_euclidean_transform(10.0 * math.pi / 180.0, [10, 20]))

def exercise8_inverse_euclidean_transform(im_org, tform):
    transformed_img_inverse = warp(im_org, tform.inverse)
    show_comparison(im_org, transformed_img_inverse, "Inverse Euclidean Transform")
#exercise8_inverse_euclidean_transform(im_org, tform)

def exercise9_similarity_transform(im_org):

    similarity_tform = SimilarityTransform(scale=0.6, rotation=15 * math.pi / 180.0, translation=(40, 30))
    similarity_transformed_img = warp(im_org, similarity_tform)
    show_comparison(im_org, similarity_transformed_img, "Similarity Transform")
#exercise9_similarity_transform(im_org)

def exercise10_swirl_transform(im_org, strength, radius, center=None):
    swirl_img = swirl(im_org, strength=strength, radius=radius, center=center)
    title = "Swirl Transform" if center is None else "Swirl Transform with Center"
    show_comparison(im_org, swirl_img, title)
c = [500, 400]
#exercise10_swirl_transform(im_org, strength=10, radius=300, center=c)

def exercise11_blend_images(src_img, dst_img):
    blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
    io.imshow(blend)
    io.show()
#exercise11_blend_images(io.imread('data/Hand1.jpg'), io.imread('data/Hand2.jpg'))

def exercise12_visualize_landmarks(src_img, src):
    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.show()
#exercise12_visualize_landmarks(io.imread('data/Hand1.jpg'), np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]]))

def exercise13_plot_landmarks(src, dst):
    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '-r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()
#exercise13_plot_landmarks(np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]]), np.array([[600, 280], [340, 190], [140, 410], [270, 540], [620, 460]]))

def exercise14_compute_alignment_error(src, dst):
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    return f
#exercise14_compute_alignment_error(np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]]), np.array([[600, 280], [340, 190], [140, 410], [270, 540], [620, 460]]))

def exercise15_optimal_euclidean_transform(src, dst):
    tform = EuclideanTransform()
    tform.estimate(src, dst)
    src_transform = matrix_transform(src, tform.params)
    exercise14_compute_alignment_error(src_transform, dst)
    fig, ax = plt.subplots()
    ax.plot(src_transform[:, 0], src_transform[:, 1], '-b', markersize=12, label="Transformed Source")
    ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks after alignment")
    plt.show()
    return tform, src_transform
#exercise15_optimal_euclidean_transform(np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]]), np.array([[600, 280], [340, 190], [140, 410], [270, 540], [620, 460]])) 

def exercise16_apply_transformation_to_image(src_img, dst_img, tform):
    warped = warp(src_img, tform.inverse)
    blend_warped = 0.5 * img_as_float(warped) + 0.5 * img_as_float(dst_img)
    io.imshow(blend_warped)
    io.show()
#exercise16_apply_transformation_to_image(io.imread('data/Hand1.jpg'), io.imread('data/Hand2.jpg'), exercise15_optimal_euclidean_transform(np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]]), np.array([[600, 280], [340, 190], [140, 410], [270, 540], [620, 460]]))[0])

def exercise17_video_transform():
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        str = math.sin(counter / 10) * 10
        transformed_frame = swirl(frame, strength=str, radius=300)
        transformed_frame = (transformed_frame * 255).astype(np.uint8)
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Transformed Video', transformed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()
exercise17_video_transform()


"""
# Exercise 1: Image rotation
im_org = io.imread('../data/NusaPenida.png')
rotation_angle = 10
rotated_img = rotate(im_org, rotation_angle)
show_comparison(im_org, rotated_img, "Rotated image")

# Exercise 2: Rotation with different center points
rot_center = [0, 0]
rotated_img_center = rotate(im_org, rotation_angle, center=rot_center)
show_comparison(im_org, rotated_img_center, "Rotated with center (0, 0)")

# Exercise 3: Rotation with different background filling modes
rotated_img_reflect = rotate(im_org, rotation_angle, mode="reflect")
rotated_img_wrap = rotate(im_org, rotation_angle, mode="wrap")
show_comparison(im_org, rotated_img_reflect, "Reflect mode")
show_comparison(im_org, rotated_img_wrap, "Wrap mode")

# Exercise 4: Rotation with constant fill value
rotated_img_constant = rotate(im_org, rotation_angle, resize=True, mode="constant", cval=1)
show_comparison(im_org, rotated_img_constant, "Constant mode with cval=1")

# Exercise 5: Automatic resizing
rotated_img_resize = rotate(im_org, rotation_angle, resize=True)
show_comparison(im_org, rotated_img_resize, "Resized image")

# Exercise 6: Euclidean transformation
rotation_angle_rad = 10.0 * math.pi / 180.0
trans = [10, 20]
tform = EuclideanTransform(rotation=rotation_angle_rad, translation=trans)
print("Euclidean Transform Matrix:\n", tform.params)

# Exercise 7: Apply Euclidean transformation
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "Euclidean Transform")

# Exercise 8: Inverse Euclidean transformation
transformed_img_inverse = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img_inverse, "Inverse Euclidean Transform")

# Exercise 9: Similarity transformation
similarity_tform = SimilarityTransform(scale=0.6, rotation=15 * math.pi / 180.0, translation=(40, 30))
similarity_transformed_img = warp(im_org, similarity_tform)
show_comparison(im_org, similarity_transformed_img, "Similarity Transform")

# Exercise 10: Swirl transformation
swirl_img = swirl(im_org, strength=10, radius=300)
show_comparison(im_org, swirl_img, "Swirl Transform")

swirl_img_center = swirl(im_org, strength=10, radius=300, center=(500, 400))
show_comparison(im_org, swirl_img_center, "Swirl Transform with Center")

# Exercise 11: Blend two images
src_img = io.imread('Hand1.jpg')
dst_img = io.imread('Hand2.jpg')
blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
io.imshow(blend)
io.show()

# Exercise 12: Visualize source landmarks
src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
plt.imshow(src_img)
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()

# Exercise 13: Place landmarks on destination image
dst = np.array([[600, 280], [340, 190], [140, 410], [270, 540], [620, 460]])
fig, ax = plt.subplots()
ax.plot(src[:, 0], src[:, 1], '-r', markersize=12, label="Source")
ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
ax.invert_yaxis()
ax.legend()
ax.set_title("Landmarks before alignment")
plt.show()

# Exercise 14: Compute alignment error
e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")

# Exercise 15: Optimal Euclidean transformation
tform = EuclideanTransform()
tform.estimate(src, dst)
src_transform = matrix_transform(src, tform.params)
fig, ax = plt.subplots()
ax.plot(src_transform[:, 0], src_transform[:, 1], '-b', markersize=12, label="Transformed Source")
ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
ax.invert_yaxis()
ax.legend()
ax.set_title("Landmarks after alignment")
plt.show()

# Compute alignment error after transformation
e_x = src_transform[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src_transform[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error after transformation F: {f}")

# Exercise 16: Apply transformation to source image
warped = warp(src_img, tform.inverse)
blend_warped = 0.5 * img_as_float(warped) + 0.5 * img_as_float(dst_img)
io.imshow(blend_warped)
io.show()

# Exercise 17: Video transformations

def video_transform():
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        str = math.sin(counter / 10) * 10
        transformed_frame = swirl(frame, strength=str, radius=300)
        transformed_frame = (transformed_frame * 255).astype(np.uint8)
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Transformed Video', transformed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()

# Uncomment the line below to run the video transformation
# video_transform()
"""