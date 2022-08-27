neighbor_vectors_3x3 = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
neighbor_weights_3X3 = [1,1,1, 1,1,1, 1,1,1]
neighbor_vectors_2X2 = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
neighbor_weights_2X2 = [0.25,0.5,0.25, 0.5,1,0.5, 0.25,0.5,0.25]
neighbor_vectors_plus = [[-1,0], [0,-1], [0,0], [0,1],[1,0]]
neighbor_weights_plus = [1, 1, 1, 1,1]

NEIGHBORHOODS = {"3x3": (neighbor_vectors_3x3,neighbor_weights_3X3)\
"2x2":(neighbor_vectors_2X2,neighbor_weights_2X2),\
"plus": (neighbor_vectors_plus,neighbor_weights_plus)}

for name in NEIGHBORHOODS:
    NEIGHBORHOODS[name][0] = np.array(NEIGHBORHOODS[name][0], dtype = np.int32)
    NEIGHBORHOODS[name][1] = np.array(NEIGHBORHOODS[name][1], dtype = np.int32)

def flatt_and_hist_mat(mat, hole_num):
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

def plot_0_pixel_as_255_for_proportion(img_arr):
    for img in img_arr:
        img[0][0] =255
    plot_imgs_arr(img_arr)

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

def separation_func(img, cuttof):
    img_copy = np.copy(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_copy[row][col] = int(img_copy[row][col]>cuttof)
    return img_copy