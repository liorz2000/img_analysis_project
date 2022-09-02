import sys
cv2_directory = 'C:\\python\\python39\\lib\\site-packages'
if cv2_directory not in sys.path:
    sys.path.append(cv2_directory)
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

PATH_C1 = "images/defective_examples/case1_{}_image.tif"
PATH_C2 = "images/defective_examples/case2_{}_image.tif"
PATH_C3 = "images/non_defective_examples/case3_{}_image.tif"
paths = [PATH_C1,PATH_C2,PATH_C3]
ins = "inspected"
ref = "reference"

def get_imgs(prev_path = "../"):
    """
    @ prev path (string): 
    the relative path from the interpreter to the images directory

    @ return (matrix of np.matrixes): 
    the 6 images as 3X2 matrix. Each row for each case.
    """
    res = []
    for i in range(3):
        res.append([])
        path_ref = paths[i].format(ref)
        path_ins = paths[i].format(ins)
        ref_img = np.array(cv2.imread(prev_path+path_ref, cv2.IMREAD_GRAYSCALE), dtype = np.int32)
        ins_img = np.array(cv2.imread(prev_path+path_ins, cv2.IMREAD_GRAYSCALE), dtype = np.int32)
        res[-1].append(ref_img)
        res[-1].append(ins_img)
    return res

def plot_imgs_matrix(imgs_mat, fs = (8,6)):
    """
    @ imgs_mat (matrix of np.matrixes): what to plot
    @ fs (int): the figure size you want to plot the picture in

    @ plot:
        imgs_mat[0][0] ... imgs_mat[0][n]
                        .
                        .
                        .
        imgs_mat[m][0] ... imgs_mat[m][n]
    """
    plt.figure(figsize = fs, dpi=80)
    rows = len(imgs_mat)
    cols = len(imgs_mat[0])
    for i in range(rows):
        for j in range(cols):
            plt.subplot(cols,rows,i+rows*j+1)
            plt.imshow(imgs_mat[i][j], cmap = 'gray')
    plt.show()

def plot_imgs_arr(imgs_arr, fs = (8,3)):
    """
    @ imgs_arr (array of np.matrixes): what to plot
    @ fs (int): the figure size you want to plot the picture in

    @ plot:
        imgs_arr[0] ... imgs_arr[n]
    """
    plt.figure(figsize = fs, dpi=80)
    rows = len(imgs_arr)
    for i in range(rows):
        plt.subplot(1,rows,i+1)
        plt.imshow(imgs_arr[i], cmap = 'gray')
    plt.show()

def plot_img(img, fs = (4,3)):
    """
    @ img (np.matrix): what to plot
    @ fs (int): the figure size you want to plot the picture in

    @ plot:
        img
    """
    plt.figure(figsize = fs, dpi=80)
    plt.imshow(img, cmap = 'gray')
    plt.show()

def get_polynom_mult(img_reference , img_inspected, is_pad = False):
    """
    @ img_reference (np.matrix): the reference image
    @ img_inspected (np.matrix): the inspected image
    @ is_pad (bool): decide if before the the polynom mult we want pad
    the images (ref image and reflection of the ins image) to 2m-1 X 2n-1
    by zeros, to avoid multiplications of opposite mergins.
    It is relevant to "pad" and "pad_and_add" methods of finding move vector.

    @ return:
        the polynom mult of ref image and reflection of the ins image.

    remark: 
    It moved by (1,1) from real mult and it will be fixed just at get_max_vector func.
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
    """
    @ imgs_mat (matrix of np.matrixes): couples of ref and ins images

    @ return:
        array of the product of their expectentions
    """
    res = []
    for case in imgs_mat:
        exp0 = np.mean(case[0])
        exp1 = np.mean(case[1])
        res.append(exp0 * exp1)
    return res

def add_margin_normalization_to_padded_mult(mult, mul_of_expectations):
    """
    @ mult (np.matrix): the zeor-padded polynom product
    @ mul of expectations (int): expec(ref) *expec(int)

    @return (np.matrix): It is relevant to the method "pad_and_add" of move_seach type.
    In the padded product, as much as you far from the center you have more multipilication 
    by zero in the sum. This function add to each of the sum the expevtation, everege mult as
    number as zero mults. And then return The new 2m-1 X 2n-1 np.matrix.
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
    """
    @ mults (array of np.matrixes): the zeor-padded polynom product
    @ mul of expectations (array of ints): expec(ref) *expec(int)

    @return (np.matrix): It is like add_margin_normalization_to_padded_mult,
    but on array of mults and mul, do the function on each pair and return
    the array of the results.
    """
    res = []
    length = len(mults)
    for i in range(length):
        res.append(add_margin_normalization_to_padded_mult(mults[i], exp_muls[i]))
    return res

def get_mults(imgs, is_pad = False):
    """
    @ imgs (matrix of np.matrixes): 
    the matrix is 2Xn and each row is couple of ref image and ins image.
    @ is_pad (bool): determine if do padding before polynom product.

    @ return (array of np.matrixes): 
    array of the polynom product for each pair. 
    """
    res = []
    for case in imgs:
        res.append(get_polynom_mult(case[0] , case[1], is_pad))
    return res

def cut_mergins(img_reference , img_inspected, move_vector = (0,0)):
    """
    @ img_reference (np.matrix): the reference image
    @ img_inspected (np.matrix): the inspected image
    @ move_vector (tuple of int): the vector of how much you need to move 

    @return  (tuple of np.metrixes):
    """
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
    """
    @ imgs (matrix of np.matrixes): 2Xn matrix of images, 
    represent n tuples of ref and inf
    @ move_vectors (array of tuples of int): array of n move vectors, 
    between each ref-ins tuple.

    @ return (matrix of np.matrixes):
    """
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
    """
    @ num (int): 
    @ mod (int):

    @return (int): number x in [-mod/2, mod/2] such that x=num (mod) 
    """
    num = num%mod
    if 2*num <= mod:
        return num
    return num - mod

def get_max_vector(pol_mult, is_pad = False):
    """
    @ pol_mult (np.matrix): the matrix of multiplication, 
    if "normal" it from size of original pricture
    if "pad" it the padded product size 2nX2m
    if "pad_and_add" it the padded but with the margin_normalization.
    @ is_pad (bool): is the product matrix padded or not.

    @ return (tuple of int): the cordinates of the move vector
    """
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
    """
    @ pol_mults (array of np.matrix): array of n product matrixes
    @ is_pad (bool): Are the products padded or not

    @ return (tuple of int): array of n move vectors
    """
    res = []
    for pol_mult in pol_mults:
        gmv = get_max_vector(pol_mult, is_pad)
        if gmv == "Failed":
            res.append("Failed")
        else:
            res.append(gmv[1])
    return res

def find_correlation(img_reference, img_inspected):
    """
    img_reference (np.matrix): the cutted reference image
    img_inspected (np.matrix): the cutted inspected image

    return (float): get the correlation between the two cutted images, 
    when you think both of them as 0-255 variable.
    """
    ref = img_reference
    ins = img_inspected
    numeretor = np.mean((ref - np.mean(ref))*(ins - np.mean(ins)))
    return numeretor / (np.std(ref) * np.std(ins))

def find_correlations(img_mat):
    """
    @ imgs_mat (matrix of np.matrixes): 2Xn matrix of images, 
    represent n tuples of ref and inf

    return (array of float): return the array of correletions.
    """
    res = []
    for case in img_mat:
        if case == "Failed":
            res.append("Failed")
        else:
            res.append(find_correlation(case[0], case[1]))
    return res


