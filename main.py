from move_find import *
from separation import *
from convexation import *

def get_dict_of_the_proccess(prev_path = "../", move_search = "normal", imgs = "the_three_given",\
	means_arr = [], cuts_r = [], cuts_a = [], conv_raduis = 3):
	"""
	@ prev_path (string):
	In case the user didn't gave pictures, 
	It is the relative path to images directory of the three images
	@ move_search (string): 
	    The algorithem to find the move vector between pictures.  
		value need to be one of those:
		"normal": just do furie
		"pad": furie with double zero paddig
		"pad_and_add": prev + adding repairs
	@ imgs (matrix of np.matrix or string):
	represent the pictures that the alorithem will work on.
	if string: it must be "the_three_given", represent
	the three cupules of ins-ref images that Spais gave.
	if matrix: then it must be 2xn, whitch are n couples
	of ins-ref images.
	@ means_arr (array of strings): 
	The names of neighberhoods type to do mean around pixel.
	For each image in diff_imgsXmean_type it calculate the 
	mean of the diff.
	@ cuts_r (array of int):
	The relative cuts To devide picture to good and defected pixels.
	For each diff_imgsXmean_typeXcut it take the abs(the mean of the diff) 
	and then separate it to pixels above max*cut and below. 
	@ cuts_a (array of int):The relative cuts To devide picture to good and bad pixels.
	For each diff_imgsXmean_typeXcut it take the abs(the mean of the diff) 
	and then separate it to pixels above cut and below. 
	@ conv_raduis (int): 
	In this stage we assume that spots are convex.
	So if there is pixel above and below that detected as defected, 
	it will be detected as defected as well. The same with right-left.
	The conv_raduis is the raduis of finding the neighbor pixels, 
	in these four directions.

	@result (dict): every detaile in the process. 

	"""
	if move_search not in ["normal", "pad" , "pad_and_add"]:
		raise Exception("No valid move_search")

	if imgs == "the_three_given":
		imgs = get_imgs(prev_path)
	if move_search == "normal":
		mults = get_mults(imgs)
		move_vectors = get_max_vectors(mults)
	else:
		mults = get_mults(imgs, True)
		if move_search == "pad_and_add":
			exp_muls = calc_muls_of_expectations(imgs)
			mults = add_margin_normalization_to_padded_mults(mults, exp_muls)
		move_vectors = get_max_vectors(mults, True)
	cutted_imgs = cut_all_mergins(imgs, move_vectors)
	diff_imgs = sub_all_imgs(cutted_imgs, False)
	diff_abs_imgs = sub_all_imgs(cutted_imgs)
	correlations = find_correlations(cutted_imgs)
	diff_means = mean_imgsXneighborhoods(diff_imgs , means_arr)
	diff_means_abs = abs_dicts_of_imgs(diff_means)
	diff_means_abs_separations = separation_dicts_of_imgs(diff_means_abs, cuts_r, cuts_a)
	diff_means_abs_separations_relative = diff_means_abs_separations[0]
	diff_means_abs_separations_absolute = diff_means_abs_separations[1]
	diff_means_abs_separations_relative_con = \
	convexation_list_dict_dict(diff_means_abs_separations_relative, conv_raduis)
	diff_means_abs_separations_absolute_con = \
	convexation_list_dict_dict(diff_means_abs_separations_absolute, conv_raduis)


	dct = {"mults":mults, "move_vectors":move_vectors}
	dct["imgs"] = imgs
	dct["cutted_imgs"] = cutted_imgs
	dct["diff_imgs"] = diff_imgs
	dct["diff_abs_imgs"] = diff_abs_imgs
	dct["correlations"] = correlations
	dct["move_search"] = move_search
	dct["diff_means"] = diff_means
	dct["diff_means_abs"] = diff_means_abs
	dct["diff_means_abs_separations_relative"] = diff_means_abs_separations_relative
	dct["diff_means_abs_separations_absolute"] = diff_means_abs_separations_absolute
	dct["diff_means_abs_separations_relative_con"] = diff_means_abs_separations_relative_con
	dct["diff_means_abs_separations_absolute_con"] = diff_means_abs_separations_absolute_con

	return dct

def solution(img1, img2):
	"""
	@img1 (np.matrix) reference image
	@img2 (np.matrix): inspected image

	@result(np.matrix): The 1-0 image that requested,
	1 for defected pixel and 0 for good.
	"""
	imgs = [[img1, img2]]
	cut = 84
	local_mean_type = "3x3"
	dct = get_dict_of_the_proccess(imgs = imgs, means_arr = [local_mean_type], cuts_a = [cut])
	return dct["diff_means_abs_separations_absolute_con"][0][local_mean_type][cut]

# The submition 4/9/22 #