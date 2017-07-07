import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from skimage import exposure
from scipy.ndimage.measurements import label

#Color feature extract
def color_hist(image, nbins=32, bins_range=[0, 256]):
    hist1 = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    hist2 = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    hist3 = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    return features

#Spatial binning
def spatial_bin(image, size=(32, 32)):
    scaled_img = cv2.resize(image, dsize=size)
    return scaled_img.ravel()

#Hog features
def get_hog_feature(image, orient=9, pix_per_cell=8, cell_per_block=2, feature_vec=True):
    """
    `image`: one channel image
    `orient`: the number of orientations of hog transform
    `pix_per_cell`: pixels per cell of hog transform
    `cell_per_block`: cells per block of hog transform
    `feature_vec`: bool, indicate whether to flatten the feature matrix
    """
    features = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block),
                    visualise=False, feature_vector=feature_vec)
    return features

def extract_features(imgs, cspace="RGB", nbins=32, bins_range=(0, 256), spatial_size=(32, 32), orient=9, pix_per_cell=8,
                    cell_per_block=2, hog_channel="ALL", color_feat=True, spatial_feat=True, hog_feat=True):
    """
    `imgs`: the file names of image
    `cspace`: the color space we want
    `nbins`: number of bins of the color histogram
    `bins_range`: the range of color histogram
    `spatial_size`: spatial binning size
    `orient`: the number of orientations of hog
    `pix_per_cell`: pixels per cell of one direction in hog
    `cell_per_block`: cells per block of one direction in hog
    `hog_channel`: the channel of image to extract hog feature, `ALL` for all channel, `0` for first channel, 
                   `1` for the second channel, `2` for the third channel
    `color_feat`: bool, whether to extract color histogram feature
    `spatial_feat`: bool, whether to extract spatial binning feature
    `hog_feat`: bool, whether to extract hog feature
    `return`: list, the extracted features for each image
    """
    features = []
    for img in imgs:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature = []
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
        if color_feat:
            color_feature = color_hist(feature_img, nbins=nbins, bins_range=bins_range)
            feature.append(color_feature)
        if spatial_feat:
            spatial_feature = spatial_bin(feature_img, size=spatial_size)
            feature.append(spatial_feature)
        if hog_feat:
            hog_feature = []
            if hog_channel == "ALL":
                for i in range(feature_img.shape[2]):
                    hog_feature.append(get_hog_feature(feature_img[:, :, i], orient=orient, pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block))
                hog_feature = np.ravel(hog_feature)
            else:
                hog_feature = get_hog_feature(feature_img[:, :, hog_channel], orient=orient, pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block)
            feature.append(hog_feature)
        feature = np.concatenate(feature)
        features.append(feature)
    return features

