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

neighbor_vectors_3x3 = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
neighbor_vectors_2X2 = [[-1,-1], [-1,0], [0,-1], [0,0]]
neighbor_vectors_plus = [[-1,0], [0,-1], [0,0], [0,1],[1,0]]
neighborhoods = {"3x3": neighbor_vectors_3x3, "2x2":neighbor_vectors_2X2, "plus": neighbor_vectors_plus}
for name in neighborhoods:
    neighborhoods[name] = np.array(neighborhoods[name])

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

def plot_imgs_matrix(imgs_mat, fs = (8,6)):
    plt.figure(figsize = fs, dpi=80)
    rows = len(imgs_mat)
    cols = len(imgs_mat[0])
    for i in range(rows):
        for j in range(cols):
            plt.subplot(cols,rows,i+rows*j+1)
            plt.imshow(imgs_mat[i][j], cmap = 'gray')
    plt.show()

def plot_imgs_arr(imgs_arr, fs = (8,3)):
    plt.figure(figsize = fs, dpi=80)
    rows = len(imgs_arr)
    for i in range(rows):
        plt.subplot(1,rows,i+1)
        plt.imshow(imgs_arr[i], cmap = 'gray')
    plt.show()

def get_polynom_mult(img_reference , img_inspected, is_pad = False):
    """
    Inputs must be 2 numpy arrays represent gray-scale 0-255
    """
    img_inspected = img_inspected[::-1,::-1]
    rows, colums = img_reference.shape
    if is_pad:
        img_reference = np.lib.pad(img_reference,((0,rows-1),(0,colums-1)))
        img_inspected = np.lib.pad(img_inspected,((0,rows-1),(0,colums-1)))
    fft_img_reference = np.fft.fft2(img_reference)
    fft_img_inspected = np.fft.fft2(img_inspected)
    fft_pol_mult = fft_img_reference * fft_img_inspected
    if is_pad:
        num_to_normalize = (2*rows - 1)*(2*colums - 1)
    else:
        num_to_normalize = rows * colums
    pol_mult = np.real(np.fft.fft2(fft_pol_mult)/num_to_normalize)
    return pol_mult

def calc_muls_of_expectations(imgs_mat):
    res = []
    for case in imgs_mat:
        exp0 = np.mean(case[0])
        exp1 = np.mean(case[1])
        res.append(exp0 * exp1)
    return res

def add_margin_normalization_to_padded_mult(mult, mul_of_expectations):
    """
    mul_of_expectations is:
    the expectection of pixel in ref *
    *  the expectection of pixel in ins
    """
    center_row = int(mult.shape[0]/2) + 1
    center_col = int(mult.shape[1]/2) + 1

    rows_num = center_row
    cols_num = center_row

    original_area = rows_num * cols_num
    for row in range(mult.shape[0]):
        for col in range(mult.shape[1]):
            row_diff = abs(center_row - row)
            col_diff = abs(center_col - col)
            common_area = (rows_num - row_diff) * (cols_num - col_diff)
            num_lost_pixels = original_area - common_area
            mult[row][col] += mul_of_expectations * num_lost_pixels
    return mult

def add_margin_normalization_to_padded_mults(mults, exp_muls):
    res = []
    length = len(mults)
    for i in range(length):
        res.append(add_margin_normalization_to_padded_mult(mults[i], exp_muls[i]))
    return res

def get_mults(imgs, is_pad = False):
    res = []
    for case in imgs:
        res.append(get_polynom_mult(case[0] , case[1], is_pad))
    return res

def cut_mergins(img_reference , img_inspected, move_vector = (0,0)):
    if move_vector == "Failed":
        return "Failed"

    row_move = move_vector[0]
    col_move = move_vector[1]

    # row cut
    if row_move > 0:
        img_reference = img_reference[:-row_move,]
        img_inspected = img_inspected[row_move:,]
    elif row_move < 0:
        img_reference = img_reference[-row_move:,]
        img_inspected = img_inspected[:row_move,]

    # colum cut
    if col_move > 0:
        img_reference = img_reference[:,:-col_move]
        img_inspected = img_inspected[:,col_move:]
    elif col_move < 0:
        img_reference = img_reference[:,-col_move:]
        img_inspected = img_inspected[:,:col_move]

    return img_reference, img_inspected

def cut_all_mergins(imgs, move_vectors = None):
    res = []
    if move_vectors is None:
        return imgs

    case_num = len(imgs) 
    for case in range(case_num):
        cutted_imgs = cut_mergins(imgs[case][0], imgs[case][1], move_vectors[case])
        if cutted_imgs == "Failed":
            res.append("Failed")
        else:
            res.append([cutted_imgs[0], cutted_imgs[1]])
    return res

def small_modulo(num, mod):
    num = num%mod
    if 2*num <= mod:
        return num
    return num - mod

def get_max_vector(pol_mult, is_pad = False):
    max_value = pol_mult[0][0]
    max_point = (0,0)
    
    for row in range(pol_mult.shape[0]):
        for col in range(pol_mult.shape[1]):
            if max_value < pol_mult[row][col]:
                max_value = pol_mult[row][col]
                max_point = (row, col)
    if is_pad:
        center_row = int(pol_mult.shape[0]/2) + 1
        center_col = int(pol_mult.shape[1]/2) + 1
        diff_row = max_point[0] - center_row
        diff_col = max_point[1] - center_col
        if abs(diff_row) > pol_mult.shape[0]/4 or abs(diff_col) > pol_mult.shape[1]/4:
            print("Failed add_and_padd return string Failed")
            return "Failed"
        return max_value, (diff_row ,diff_col)
    
    small_row_diff = small_modulo(max_point[0],pol_mult.shape[0])
    small_col_diff = small_modulo(max_point[1],pol_mult.shape[1])
    return max_value, (small_row_diff - 1,small_col_diff - 1)

def get_max_vectors(pol_mults, is_pad = False):
    res = []
    for pol_mult in pol_mults:
        gmv = get_max_vector(pol_mult, is_pad)
        if gmv == "Failed":
            res.append("Failed")
        else:
            res.append(gmv[1])
    return res

def sub_imgs(img_reference , img_inspected):
    return abs(img_reference - img_inspected)

def sub_all_imgs(imgs):
    res = []
    for case in imgs:
        if case == "Failed":
            res.append("Failed")
        else:
            res.append(sub_imgs(case[0] , case[1]))
    return res

def plot_0_pixel_as_255_for_proportion(img_arr):
    for img in img_arr:
        img[0][0] =255
    plot_imgs_arr(imgs_arr)

def find_correlation(img_reference, img_inspected):
    ref = img_reference
    ins = img_inspected
    numeretor = np.mean((ref - np.mean(ref))*(ins - np.mean(ins)))
    return numeretor / (np.std(ref) * np.std(ins))

def find_correlations(img_mat):
    res = []
    for case in img_mat:
        if case == "Failed":
            res.append("Failed")
        else:
            res.append(find_correlation(case[0], case[1]))
    return res

# def neighborhood_mean(img, neighborhood):
#   nrow, ncol = img.shape
#   for row in range(nr)
