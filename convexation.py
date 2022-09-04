import sys
cv2_directory = 'C:\\python\\python39\\lib\\site-packages'
if cv2_directory not in sys.path:
    sys.path.append(cv2_directory)
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import product

from separation import *

def convexation(img, raduis):
	"""
	@img (np.matrix):
	Image of ones and zeros, represent if pixel is defected.
	@raduis (int):
	We assume that spots are convex.
	So if there is pixel above and below that detected as defected, 
	it will be detected as defected as well. The same with right-left.
	It is the raduis of finding the neighbor pixels, in these four directions.

	@result (np.matrix):
	The image after convexetion accord to the raduis.
	"""
    down = np.zeros(img.shape)
    up = np.zeros(img.shape)
    right = np.zeros(img.shape)
    left = np.zeros(img.shape)
    
    for i in range(1, raduis+1):
        down[i][0] = 1
        up[-i][0] = 1
        right[0][i] = 1
        left[0][-i] = 1
    
    is_down = mean_two_mat(img,down)
    is_up = mean_two_mat(img,up)
    is_right = mean_two_mat(img,right)
    is_left = mean_two_mat(img,left)
    
    img_copy =np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if not img[row][col]:
                if is_down[row][col]:
                    if is_up[row][col]:
                        img_copy[row][col] = 1
                if is_right[row][col]:
                    if is_right[row][col]:
                        img_copy[row][col] = 1
    return img_copy

def convexation_list_dict_dict(ldd, raduis):
	"""
	@ ldd (list of dict of dict of np.matrix):
	list represent diffrent tuples of ref-ins.
	dict represent diffrent means.
	dict represent diffrent cuts. 
	@ raduis (int): read the explenation at the above func.

	@ result (list of dict of dict of np.matrix):
	Activate convexation on each np.matrix.
	"""
    for case in ldd:
    	for local_mean_type in case:
    		for cut in case[local_mean_type]:
    			case[local_mean_type][cut] = convexation(case[local_mean_type][cut], raduis)
    return ldd