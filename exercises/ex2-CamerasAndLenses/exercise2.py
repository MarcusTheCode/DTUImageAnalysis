from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import math

#Exercise 1
# The angle of a right triangle is given by tan^-1(opposite/adjacent)
def calculateAngleOfTriangle(opposite, adjacent):
    return math.degrees(math.atan(opposite/adjacent))
#print(calculateAngleOfTriangle(3, 10))

#Exercise 2
# When the object is very close to the lens, the CCD (image plane) is farther from the lens.
# As the object moves further away, the image distance decreases.
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    return -1*((f*g)/(f-g))
print(camera_b_distance(5, 3000))
#print(camera_b_distance(15, 1000))
#print(camera_b_distance(15, 5000))
#print(camera_b_distance(15, 15000))


