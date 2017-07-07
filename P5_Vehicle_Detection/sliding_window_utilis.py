import numpy as np
import cv2
from feature_extraction_utilis import color_hist, spatial_bin, get_hog_feature
from scipy.ndimage.measurements import label


def find_car(image, clf, X_Scaler, y_start, y_stop, scale=1, cspace="RGB", nbins=32, bins_range=(0, 256), spatial_size=(32, 32),
            orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL", color_feat=True, spatial_feat=True, hog_feat=True):
    """
    `return`: image which has been drawn box, windows where the car have been found
    """
    draw_img = np.copy(image)
    windows = []
    image = image[y_start:y_stop, :, :]
    if scale != 1:
        image = cv2.resize(image, (np.int(image.shape[1]/scale), np.int(image.shape[0]/scale)))
    if cspace != "RGB":
        if cspace == "HSV":
            feature_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == "HLS":
            feature_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == "YUV":
            feature_img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == "LUV":
            feature_img = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == "YCrCb":
            feature_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_img = np.copy(image)
    #First calculate the hog of the whole image
    image_hogs = []
    if hog_channel == "ALL":
        hog1 = get_hog_feature(feature_img[:,:,0], orient=orient, pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block, feature_vec=False)
        hog2 = get_hog_feature(feature_img[:,:,1], orient=orient, pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block, feature_vec=False)
        hog3 = get_hog_feature(feature_img[:,:,1], orient=orient, pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block, feature_vec=False)
        image_hogs.append(hog1)
        image_hogs.append(hog2)
        image_hogs.append(hog3)
    else:
        hog = get_hog_feature(feature_img[:,:,hog_channel], orient=orient, pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block, feature_vec=False)
        image_hogs.append(hog)
    window = 64
    cell_per_window = window//pix_per_cell
    block_per_window = cell_per_window - cell_per_block + 1
    cell_per_step = 2
    nxsteps = ((image.shape[1] - window)//pix_per_cell)//cell_per_step
    nysteps = ((image.shape[0] - window)//pix_per_cell)//cell_per_step
    for nx in range(nxsteps):
        for ny in range(nysteps):
            feature = []
            cell_x = nx*cell_per_step
            cell_y = ny*cell_per_step
            left_upper_corner = (cell_x*pix_per_cell, cell_y*pix_per_cell)
            sub_image = cv2.resize(feature_img[left_upper_corner[1]:left_upper_corner[1]+window, 
                                               left_upper_corner[0]:left_upper_corner[0]+window], (64, 64))
            if color_feat:
                color_feature = color_hist(sub_image, nbins=nbins, 
                                                       bins_range=bins_range)
                feature.append(color_feature)
            if spatial_feat:
                spatial_feature = spatial_bin(sub_image, spatial_size)
                feature.append(spatial_feature)
            if hog_feat:
                hog_feature = []
                for image_hog in image_hogs:
                    hog_feature.append(image_hog[cell_y:cell_y+block_per_window, cell_x:cell_x+block_per_window].ravel())
                hog_feature = np.concatenate(hog_feature)
                feature.append(hog_feature)
            feature = np.concatenate(feature).reshape(1, -1)
            normalized_feature = X_Scaler.transform(feature)
            pred = clf.predict(normalized_feature)
            if pred == 1:
                left_upper_corner = (np.int(cell_x*pix_per_cell*scale), np.int(cell_y*pix_per_cell*scale)+y_start)
                right_bottom_corner = (np.int((cell_x*pix_per_cell+window)*scale), np.int((cell_y*pix_per_cell+window)*scale)+y_start)
                cv2.rectangle(draw_img, left_upper_corner, right_bottom_corner, (0, 0, 255), 6)
                windows.append((left_upper_corner, right_bottom_corner))
    return draw_img, windows

def make_heatmap(image, windows):
    """
    This function is to make heatmap according the windows
    `windows`: list of tuples, containing the coordinates of the rectangles
    `image`: the heatmap with the same size as image
    `return`: the heatmap with only one channel
    """
    heatmap = np.zeros_like(image[:, :, 0])
    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return heatmap  

def heatmap_threshold(heatmap, threshold):
    """
    This function is to threshold the heatmap.
    `heatmap`: the input heatmap to be thresholded
    `threshold`: the threshold of the heatmap
    `return`: the heatmap after thresholding
    """   
    heatmap[heatmap <= threshold] = 0
    return heatmap

def get_heatmap_label(heatmap):
    """
    This function is to label the heatmap.
    """
    labeled_array, num_labels = label(heatmap)
    return labeled_array, num_labels

def draw_labeled_heatmap(image, windows, threshold):
    heatmap = make_heatmap(image, windows)
    heatmap = heatmap_threshold(heatmap, threshold)
    labeled_array, num_labels = get_heatmap_label(heatmap)
    draw_img = np.copy(image)
    for i in range(1, num_labels+1):
        nonzeros = (labeled_array == i).nonzero()
        nonzerox = nonzeros[1]
        nonzeroy = nonzeros[0]
        cv2.rectangle(draw_img, (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)), (0, 0, 255), 6)
    return draw_img
