from help_funcs import *

def get_dict_of_the_proccess(prev_path = "../", move_search = "normal", imgs = "the_three_given"):
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

	dct = {"mults":mults, "move_vectors":move_vectors}
	dct["cutted_imgs"] = cutted_imgs
	dct["diff_imgs"] = diff_imgs
	dct["diff_abs_imgs"] = diff_abs_imgs
	dct["correlations"] = correlations
	dct["move_search"] = move_search
	return dct
