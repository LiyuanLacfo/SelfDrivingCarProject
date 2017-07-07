from feature_extraction_utilis import color_hist, spatial_bin, get_hog_feature, extract_features
import get_image_file_names_utilis as get_image_names
from classifier_utilis import get_features_and_labels, normalize, train_valid_test_data, svm_classifier
import cv2
import pickle
from sliding_window_utilis import find_car, make_heatmap, heatmap_threshold, draw_labeled_heatmap
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

cspace = "RGB"
nbins = 32
bins_range = (0, 256)
spatial_size = (32, 32)
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
color_feat = True
spatial_feat = True
hog_feat = True


cars = get_image_names.get_car_names()
not_cars = get_image_names.get_non_car_names()
car_features = extract_features(cars, cspace=cspace, nbins=nbins, bins_range=bins_range, spatial_size=spatial_size, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, color_feat=color_feat,
                                  spatial_feat=spatial_feat, hog_feat=hog_feat)
not_car_features = extract_features(not_cars, cspace=cspace, nbins=nbins, bins_range=bins_range, spatial_size=spatial_size, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, color_feat=color_feat,
                                  spatial_feat=spatial_feat, hog_feat=hog_feat)

with open("features_RGB.p", "wb") as f:
    pickle.dump({
        "car_features": car_features,
        "not_car_features": not_car_features
        }, f)


with open("features_RGB.p", "rb") as f:
    data = pickle.load(f)

car_features = data["car_features"]
not_car_features = data["not_car_features"]

features, labels = get_features_and_labels(car_features, not_car_features)
scaler = normalize(features)
with open("scaler_RGB.p", "wb") as f:
    pickle.dump({
        "scaler": scaler
        }, f)

normalized_features = scaler.transform(features)
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_data(normalized_features, labels)

with open("train_data_RGB.p", "wb") as f:
    pickle.dump({
        "train_features": X_train,
        "train_labels": y_train,
        }, f)

with open("valid_data_RGB.p", "wb") as f:
    pickle.dump({
        "valid_features": X_valid,
        "valid_labels": y_valid,
        }, f)

with open("test_data_RGB.p", "wb") as f:
    pickle.dump({
        "test_features": X_test,
        "test_labels": y_test,
        }, f)

with open("train_data_RGB.p", "rb") as f:
    train_data = pickle.load(f)

with open("valid_data_RGB.p", "rb") as f:
    valid_data = pickle.load(f)
    
with open("test_data_RGB.p", "rb") as f:
    test_data = pickle.load(f)

X_train, y_train = train_data["train_features"], train_data["train_labels"]
X_valid, y_valid = valid_data["valid_features"], valid_data["valid_labels"]
X_test, y_test = test_data["test_features"], test_data["test_labels"]

classifier = svm_classifier(X_train, y_train, kernel="linear", C=3.0)
with open("clf_linear_RGB.p", "wb") as f:
    pickle.dump({
        "clf": classifier
        }, f)
valid_acc = classifier.score(X_valid, y_valid)
test_acc = classifier.score(X_test, y_test)
