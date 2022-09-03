from move_find import *
from separation import *

def get_dict_of_the_proccess(prev_path = "../", move_search = "normal", imgs = "the_three_given",\
	means_arr = [], cuts_r = [], cuts_a = []):
	"""
	type value need to be one of those:
		"normal": just do furie
		"pad": furie with double zero paddig
		"pad_and_add": prev + adding repairs
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

	return dct

def solution(img1, img2):
	imgs = [[img1, img2]]
	cut = 84
	local_mean_type = "3x3"
	dct = get_dict_of_the_proccess(imgs = imgs, means_arr = [local_mean_type], cuts_a = [cut])
	return dct['diff_means_abs_separations_absolute'][0][local_mean_type][cut]
