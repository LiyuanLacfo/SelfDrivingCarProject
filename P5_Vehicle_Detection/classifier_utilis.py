from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def get_features_and_labels(car_features, not_car_features):
    """
    `car_features`: list, the features of car images after color histogram, spatial binning and hog transform.
    `not_car_features`: list, the features of non-car images after color histogram, spatial binning and hog transform.
    `return`: the features and labels of all data
    """
    features = np.vstack((car_features, not_car_features)).astype(np.float64)
    labels = np.concatenate((np.ones(len(car_features)), np.zeros(len(not_car_features))))
    features, labels = shuffle(features, labels)
    return features, labels

def normalize(features):
    """
    This function is to do standard scaler to the features such that they have 0 mean and unit variance
    `features`: the features of the images
    `return`: the features after normalizing
    """
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def train_valid_test_data(features, labels):
    """
    This function is to make the train, valid and test data set
    `features`: the features of data
    `labels`: the labels of data
    `return`: train, validation and test data
    """
    rand_ind_1 = np.random.randint(0, 100)
    rand_ind_2 = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=rand_ind_1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=rand_ind_2)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def svm_classifier(train_features, train_labels, kernel="linear", C=1.0):
    """
    This function is to train the SVM classifier.
    `train_features`: train features
    `train_labels`: train labels
    `return`: the svm classifier
    """
    if kernel == "linear":
        clf = LinearSVC(C=C)
    elif kernel == "rbf":
        clf = SVC(kernel="rbf", C=C)
    clf.fit(train_features, train_labels)
    return clf