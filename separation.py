import sys
cv2_directory = 'C:\\python\\python39\\lib\\site-packages'
if cv2_directory not in sys.path:
    sys.path.append(cv2_directory)
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import product

neighbor_vectors_3x3 = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
neighbor_weights_3X3 = [1,1,1, 1,1,1, 1,1,1]
neighbor_vectors_2X2 = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
neighbor_weights_2X2 = [0.25,0.5,0.25, 0.5,1,0.5, 0.25,0.5,0.25]
neighbor_vectors_plus = [[-1,0], [0,-1], [0,0], [0,1],[1,0]]
neighbor_weights_plus = [1, 1, 1, 1,1]


NEIGHBORHOODS = {"3x3": [neighbor_vectors_3x3,neighbor_weights_3X3],\
"2x2":[neighbor_vectors_2X2,neighbor_weights_2X2],\
"plus": [neighbor_vectors_plus,neighbor_weights_plus]}

for name in NEIGHBORHOODS:
    NEIGHBORHOODS[name][0] = np.array(NEIGHBORHOODS[name][0], dtype = np.int32)
    NEIGHBORHOODS[name][1] = np.array(NEIGHBORHOODS[name][1], dtype = np.float64)

def flatt_and_hist_mat(mat, hole_num):
    """
    @mat (np.matrix):
    @hole_num (int):

    @plot: histogram of mat value.

    @return: the histogram of mat value.
    """
    arr = np.ravel(mat)
    plt.hist(arr, hole_num)
    plt.show()
    return np.mean(arr) , np.std(arr)

def hist_sub(img_reference , img_inspected, hole_num, do_abs = True):
    sub = sub_imgs(img_reference , img_inspected , do_abs)
    return flatt_and_hist_mat(sub, hole_num)

def sub_imgs(img_reference , img_inspected, do_abs = True):
    sub = img_reference - img_inspected
    if do_abs:
        return abs(sub)
    return sub

def sub_all_imgs(imgs, do_abs = True):
    res = []
    for case in imgs:
        if case == "Failed":
            res.append("Failed")
        else:
            res.append(sub_imgs(case[0] , case[1], do_abs))
    return res

def neighborhood_mean_old(img, neighborhood_key):
    neighborhood = NEIGHBORHOODS[neighborhood_key][0]
    nrow, ncol = img.shape
    numerator = np.zeros((nrow, ncol))
    denumerator = np.zeros((nrow, ncol))
    for row in range(nrow):
        for col in range(ncol):
            for neighbor in neighborhood:
                if 0<=row+neighbor[0]<nrow:
                    if 0<=col+neighbor[1]<ncol:
                        denumerator[row][col] += 1
                        numerator[row][col] += img[row+neighbor[0]][col+neighbor[1]]
    return numerator/denumerator

####################################################################
                        # neighborhood_mean_fft #
####################################################################
def rotate(a, how_much):
    a_copy = np.copy(a)
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            a_copy[row,col] = a[(row+how_much)%a.shape[0], (col+how_much)%a.shape[1]]
    return a_copy

def mult_a_b(a,b):
    a = a[::-1,::-1]
    a = rotate(a, -1)
    b = b[::-1,::-1]
    b = rotate(b, -1)
    
    a = np.fft.fft2(a)
    b = np.fft.fft2(b)
    c = a*b
    c= np.fft.fft2(c)
    return np.round(np.real(c)/(c.shape[0] *c.shape[1] ))
    
def mean_two_mat(a,b):
    b = b[::-1,::-1]
    b = rotate(b, -1)
    return mult_a_b(a,b)

def neighborhood_mean_fft(img, neighborhood_key):
    neighborhood = NEIGHBORHOODS[neighborhood_key]
    neis = neighborhood[0]
    weights =  neighborhood[1]
    
    mat_to_mult = np.zeros(img.shape)
    for i in range(len(neis)):
        mat_to_mult[neis[i][0], neis[i][1]] = weights[i]
    return mean_two_mat(img,mat_to_mult)

####################################################################
def mean_imgsXneighborhoods(img_arr, neighborhood_keys):
    res = []
    for img in img_arr:
        res.append({})
        for key in neighborhood_keys:
            res[-1][key] = neighborhood_mean_fft(img, key)
    return res

def abs_dicts_of_imgs(img_list_dicts):
    res = []
    for i in range(len(img_list_dicts)):
        res.append({})
        for key in img_list_dicts[i]:
            res[-1][key] = abs(img_list_dicts[i][key])
    return res


def separation_func(img, cuttof):
    max_img = max(img)
    img_copy = np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_copy[row][col] = int(img_copy[row][col]>cuttof*max_img)
    return img_copy

def separation_func(img, cuttof):
    max_img = max([max(row) for row in img])
    img_copy = np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_copy[row][col] = int(img_copy[row][col]>cuttof*max_img)
    return img_copy

def separation_dicts_of_imgs(img_list_dicts, cuts = []):
    res = []
    for i in range(len(img_list_dicts)):
        res.append({})
        for key in img_list_dicts[i]:
            res[-1][key] = {}
            for cut in cuts:
                res[-1][key][cut] = separation_func(img_list_dicts[i][key], cut)
    return res