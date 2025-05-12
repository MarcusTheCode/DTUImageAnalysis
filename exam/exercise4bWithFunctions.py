import os
from skimage import io, color
from skimage.morphology import disk
from combined import *

# Base data folder path
data_folder = "data/open_close/"

def exercise1():
    image = io.imread(data_folder + "lego_5.png")
    gray = color.rgb2gray(image)
    binary = binarize_image_gray(gray)
    plot_comparison(image, binary, "Binary image")

def exercise2():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    for r in [2, 5, 10]:
        eroded = apply_erosion(binary, r)
        plot_comparison(binary, eroded, f"Erosion (r={r})")

def exercise3():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    for r in [2, 5, 10]:
        dilated = apply_dilation(binary, r)
        plot_comparison(binary, dilated, f"Dilation (r={r})")

def exercise4():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    for r in [2, 5, 10]:
        opened = apply_opening(binary, r)
        plot_comparison(binary, opened, f"Opening (r={r})")

def exercise5():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    for r in [2, 5, 10]:
        closed = apply_closing(binary, r)
        plot_comparison(binary, closed, f"Closing (r={r})")

def exercise6():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    outline = compute_outline(binary, radius=1)
    plot_comparison(binary, outline, "Outline")

def exercise7():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_5.png"))
    opened = apply_opening(binary, 1)
    closed = apply_closing(opened, 15)
    outline = compute_outline(closed, radius=1)
    plot_comparison(binary, outline, "Opening + Closing + Outline")

def exercise8():
    path = data_folder + "lego_7.png"
    original = io.imread(path)
    gray = color.rgb2gray(original)
    binary = binarize_image_gray(gray)
    outline = compute_outline(binary, radius=1)
    plot_comparison(binary, outline, "Outline (lego_7)")

def exercise9():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_7.png"))
    for r in [5, 10, 15]:
        closed = apply_closing(binary, r)
        outline = compute_outline(closed, radius=1)
        plot_comparison(binary, outline, f"Closing (r={r}) + Outline")

def exercise10():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_3.png"))
    closed = apply_closing(binary, 10)
    outline = compute_outline(closed, radius=1)
    plot_comparison(binary, outline, "lego_3 Outline")

def exercise11():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_9.png"))
    outline = compute_outline(binary, radius=1)
    plot_comparison(binary, outline, "lego_9 Outline")

def exercise12():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_9.png"))
    closed = apply_closing(binary, 5)
    outline = compute_outline(closed, radius=1)
    plot_comparison(binary, outline, "Closed + Outline (lego_9)")

def exercise13():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_9.png"))
    closed = apply_closing(binary, 5)
    for r in [10, 25, 50, 100]:
        eroded = apply_erosion(closed, r)
        plot_comparison(closed, eroded, f"Erosion r={r} (lego_9)")

def exercise14():
    binary = binarize_image_gray(load_image_gray(data_folder + "lego_9.png"))
    closed = apply_closing(binary, 5)
    eroded = apply_erosion(closed, 100)
    for r in [10, 20, 30]:
        dilated = apply_dilation(eroded, r)
        plot_comparison(eroded, dilated, f"Dilation r={r} after erosion")

def exercise15():
    binary = binarize_image_gray(load_image_gray(data_folder + "puzzle_pieces.png"))
    plot_comparison(binary, binary, "Puzzle Pieces Binary")

def exercise16():
    binary = binarize_image_gray(load_image_gray(data_folder + "puzzle_pieces.png"))
    opened = apply_opening(binary, 10)
    outline = compute_outline(opened, radius=1)
    plot_comparison(binary, outline, "Opened + Outline (Puzzle)")

def main():
    print("Running Exercise 1...")
    exercise1()
    print("Running Exercise 2...")
    exercise2()
    print("Running Exercise 3...")
    exercise3()
    print("Running Exercise 4...")
    exercise4()
    print("Running Exercise 5...")
    exercise5()
    print("Running Exercise 6...")
    exercise6()
    print("Running Exercise 7...")
    exercise7()
    print("Running Exercise 8...")
    exercise8()
    print("Running Exercise 9...")
    exercise9()
    print("Running Exercise 10...")
    exercise10()
    print("Running Exercise 11...")
    exercise11()
    print("Running Exercise 12...")
    exercise12()
    print("Running Exercise 13...")
    exercise13()
    print("Running Exercise 14...")
    exercise14()
    print("Running Exercise 15...")
    exercise15()
    print("Running Exercise 16...")
    exercise16()

if __name__ == "__main__":
    main()
