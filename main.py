import sys
cv2_directory = 'C:\\python\\python39\\lib\\site-packages'
if cv2_directory not in sys.path:
    sys.path.append(cv2_directory)
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_polynom_mult(img_reference , img_inspected):
    """
    inputs must be 2 numpy arrays represent gray-scale 0-255
    """
    img_inspected = img_inspected[::-1,::-1]
    rows, colums = img_reference.shape
    img_reference = np.lib.pad(img_reference,((0,rows-1),(0,colums-1)))
    img_inspected = np.lib.pad(img_inspected,((0,rows-1),(0,colums-1)))
    fft_img_reference = np.fft.fft2(img_reference)
    fft_img_inspected = np.fft.fft2(img_inspected)
    fft_pol_mult = fft_img_reference * fft_img_inspected
    num_to_normalize = (2*rows - 1)*(2*colums - 1)
    pol_mult = np.real(np.fft.fft2(fft_pol_mult)/num_to_normalize)
    return pol_mult