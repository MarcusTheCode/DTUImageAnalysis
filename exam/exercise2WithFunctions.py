import math
from combined import (
    calculate_angle_with_atan2, solve_thin_lens, projected_image_height,
    field_of_view, pixel_size_mm, image_height_in_pixels
)

def run_exercise1():
    a = 3
    b = 10
    theta_deg = calculate_angle_with_atan2(a, b)
    print(f"Exercise 1:\nThe angle θ is {theta_deg:.2f} degrees")

def run_exercise2():
    f_mm = 15
    g_values = [0.1, 1, 5, 15]  # in meters
    print("Exercise 2:")
    for g_m in g_values:
        b_mm = solve_thin_lens(f=f_mm, g=g_m * 1000, b=None)
        print(f"  Object distance g = {g_m} m → Image distance b = {b_mm:.2f} mm")

def run_exercise4():
    object_height_m = 1.8
    object_distance_m = 5
    focal_length_mm = 5
    height_mm = projected_image_height(object_height_m, object_distance_m, focal_length_mm)
    print(f"Exercise 4:\nThomas appears {height_mm:.2f} mm tall on the CCD")

def run_exercise5():
    ccd_width_mm = 6.4
    ccd_height_mm = 4.8
    focal_length_mm = 5
    h_fov, v_fov = field_of_view(ccd_width_mm, ccd_height_mm, focal_length_mm)
    print(f"Exercise 5:\nHorizontal FOV: {h_fov:.2f}°, Vertical FOV: {v_fov:.2f}°")

def run_exercise6():
    ccd_width_mm = 6.4
    ccd_height_mm = 4.8
    pixel_width = 640
    pixel_height = 480
    height_mm = 1.8 / 5 * 5  # Uses previous projection method for consistency
    px_size_w, px_size_h = pixel_size_mm(ccd_width_mm, ccd_height_mm, pixel_width, pixel_height)
    height_px = image_height_in_pixels(height_mm, px_size_h)
    print(f"Exercise 6:\nPixel size: ({px_size_w:.4f}, {px_size_h:.4f}) mm\nThomas height on CCD: {height_px:.2f} pixels")

run_exercise2()