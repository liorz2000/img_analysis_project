import sys
cv2_directory = 'C:\\python\\python39\\lib\\site-packages'
if cv2_directory not in sys.path:
    sys.path.append(cv2_directory)
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

PATH_C1 = "../images/defective_examples/case1_{}_image.tif"
PATH_C2 = "../images/defective_examples/case2_{}_image.tif"
PATH_C3 = "../images/non_defective_examples/case3_{}_image.tif"
paths = [PATH_C1,PATH_C2,PATH_C3]
ins = "inspected"
ref = "reference"

def get_imgs():
	"""
	Get the 6 images as 3X2 matrix.
	Each row for each case.	
	"""
    res = []
    for i in range(3):
        res.append([])
        path_ins = paths[i].format(ins)
        path_ref = paths[i].format(ref)
        res[-1].append(np.array(cv2.imread(path_ref, cv2.IMREAD_GRAYSCALE)))
        res[-1].append(np.array(cv2.imread(path_ins, cv2.IMREAD_GRAYSCALE)))
    return res

def get_polynom_mult(img_reference , img_inspected):
    """
    Inputs must be 2 numpy arrays represent gray-scale 0-255
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

def get_max_vector(pol_mult):
    max_value = pol_mult[0][0]
    max_point = (0,0)
    
    for row in range(pol_mult.shape[0]):
        for col in range(pol_mult.shape[1]):
            if max_value < pol_mult[row][col]:
                max_value = pol_mult[row][col]
                max_point = (row, col)
    
    center_row = int(pol_mult.shape[0]/2)+1
    center_col = int(pol_mult.shape[1]/2)+1
    
    return max_value, (max_point[0] - center_row ,max_point[1] - center_col)