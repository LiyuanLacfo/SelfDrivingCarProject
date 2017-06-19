###Part 1 load and save images data
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle

#read and load the images from the paths in csv file
def process(file_paths, augment=False):
    lines = []
    for file in file_paths:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    images = []
    measurements = []
    for line in lines:
        image_source = line[0]
        image = cv2.imread(image_source)
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)
    assert len(images) == len(measurements)
    if augment:
        n = len(images)
        for i in range(n):
            aug_image = cv2.flip(images[i], 1)
            aug_measurement = measurements[i]*-1
            images.append(aug_image)
            measurements.append(aug_measurement)
    images = np.array(images)
    measurements = np.array(measurements)
    return images, measurements

file_path = ["./Sim_data_track_1_center/driving_log.csv", "./Sim_data_track_1_go_back/driving_log.csv",
             "./Sim_data_track_1_smooth/driving_log.csv", "./Sim_data_track_1_corner/driving_log.csv"]

train_X, train_y = process(file_path, augment=True)
train_features, val_features, train_labels, val_labels = train_test_split(train_X, train_y, test_size=0.25)

n_train = train_features.shape[0]
n_val = val_features.shape[0]

#Save the data into pickle file
train_path_1 = "train_data_1.p"
train_path_2 = "train_data_2.p"
train_path_3 = "train_data_3.p"
train_path_4 = "train_data_4.p"
train_path_5 = "train_data_5.p"
val_path_1 = "val_data_1.p"
val_path_2 = "val_data_2.p"

#save training data
if not os.path.isfile(train_path_1):
    with open(train_path_1, "wb") as f:
        pickle.dump({
            "features": train_features[:10000],
            "labels": train_labels[:10000]
        }, f)
if not os.path.isfile(train_path_2):
    with open(train_path_2, "wb") as f:
        pickle.dump({
            "features": train_features[10000:20000],
            "labels": train_labels[10000:20000]
        }, f)
if not os.path.isfile(train_path_3):
    with open(train_path_3, "wb") as f:
        pickle.dump({
            "features": train_features[20000:30000],
            "labels": train_labels[20000:30000]
        }, f)
if not os.path.isfile(train_path_4):
    with open(train_path_4, "wb") as f:
        pickle.dump({
            "features": train_features[30000:40000],
            "labels": train_labels[30000:40000]
        }, f)
if not os.path.isfile(train_path_5):
    with open(train_path_5, "wb") as f:
        pickle.dump({
            "features": train_features[40000:],
            "labels": train_labels[40000:]
        }, f)

#save validation data
if not os.path.isfile(val_path_1):
    with open(val_path_1, "wb") as f:
        pickle.dump({
            "features": val_features[:8000],
            "labels": val_labels[:8000]
        }, f)
if not os.path.isfile(val_path_2):
    with open(val_path_2, "wb") as f:
        pickle.dump({
            "features": val_features[8000:],
            "labels": val_labels[8000:]
        }, f)


### Part 2, train the model
import keras
from keras.models import Sequential
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.layers.core import Activation, Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#Define generator
def generator(sample_files, batch_size):
    n = len(sample_files)
    while 1:
        for i in range(n):
            with open(sample_files[i], "rb") as f:
                data = pickle.load(f)
            features = data["features"]
            labels = data["labels"]
            features, labels = shuffle(features, labels)
            n_samples = features.shape[0]
            for start in range(0, n_samples, batch_size):
                end = min(start+batch_size, n_samples)
                yield (features[start:end], labels[start:end])

#Define some hyperparameters
batch_size=128
epochs = 20
learning_rate= 0.0005

#Create training generator and validation generator
train_generator = generator([train_path_1, train_path_2, train_path_3, train_path_4], batch_size)
val_generator = generator([val_path_1, val_path_2], batch_size)

#Define the hyperparameters about the model structure
conv_output = [36, 48, 64, 128, 128, 128]
conv_filter = [3, 3, 3, 3, 3, 3]
conv_stride = [1, 1, 1, 1, 1, 1]
conv_padding = ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME"]
pool_filter = [2, 2, 2, 2, 2, 2]
pool_stride = [2, 2, 2, 2, 2, 2]
pool_padding = ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME"]
fc_output = [1024, 100, 50, 10]

#Build the model
model= Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
#First ConvLayer
model.add(Conv2D(conv_output[0], kernel_size=[conv_filter[0], conv_filter[0]], 
                 strides=[conv_stride[0], conv_stride[0]], padding=conv_padding[0], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[0], pool_filter[0]), 
                       strides=(pool_stride[0], pool_stride[0]), padding=pool_padding[0]))
model.add(Activation("elu"))
model.add(Dropout(0.1))
#Second conv layer
model.add(Conv2D(conv_output[1], kernel_size=[conv_filter[1], conv_filter[1]], 
                 strides=[conv_stride[1], conv_stride[1]], padding=conv_padding[1], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[1], pool_filter[1]), 
                       strides=(pool_stride[1], pool_stride[1]), padding=pool_padding[1]))
model.add(Activation("elu"))
model.add(Dropout(0.1))
#Third conv layer
model.add(Conv2D(conv_output[2], kernel_size=[conv_filter[2], conv_filter[2]], 
                 strides=[conv_stride[2], conv_stride[2]], padding=conv_padding[2], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[2], pool_filter[2]), 
                       strides=(pool_stride[2], pool_stride[2]), padding=pool_padding[2]))
model.add(Activation("elu"))
model.add(Dropout(0.1))
#Fourth Conv layer
model.add(Conv2D(conv_output[3], kernel_size=[conv_filter[3], conv_filter[3]], 
                 strides=[conv_stride[3], conv_stride[3]], padding=conv_padding[3], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[3], pool_filter[3]), 
                       strides=(pool_stride[3], pool_stride[3]), padding=pool_padding[3]))
model.add(Activation("elu"))
model.add(Dropout(0.1))

#Fifth conv layer
model.add(Conv2D(conv_output[4], kernel_size=[conv_filter[4], conv_filter[4]], 
                 strides=[conv_stride[4], conv_stride[4]], padding=conv_padding[4], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[4], pool_filter[4]), 
                       strides=(pool_stride[4], pool_stride[4]), padding=pool_padding[4]))
model.add(Activation("elu"))
model.add(Dropout(0.1))

#Sixth conv layer
model.add(Conv2D(conv_output[5], kernel_size=[conv_filter[5], conv_filter[5]], 
                 strides=[conv_stride[5], conv_stride[5]], padding=conv_padding[5], 
                 kernel_initializer="random_normal", bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(pool_filter[5], pool_filter[5]), 
                       strides=(pool_stride[5], pool_stride[5]), padding=pool_padding[5]))
model.add(Activation("elu"))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(fc_output[0]))
model.add(Activation("elu")) 
model.add(Dropout(0.5))
model.add(Dense(fc_output[1]))
model.add(Activation("elu"))
model.add(Dropout(0.5))
model.add(Dense(fc_output[2]))
model.add(Activation("elu"))
model.add(Dropout(0.5))
model.add(Dense(fc_output[3]))
model.add(Activation("elu"))
model.add(Dropout(0.5))
model.add(Dense(1))

#Define the optimizer
rms = keras.optimizers.RMSprop(lr=learning_rate)

#Compile the model
model.compile(optimizer=rms, loss="mse")

#Train the model
steps_per_epoch = int(n_train/batch_size)
steps_val = int(n_val/batch_size)

model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, verbose=1, 
                    validation_data = val_generator, validation_steps=steps_val)

#Load the test data and evaluate
with open(test_path, "rb") as f:
    test = pickle.load(f)
    test_features = test["features"]
    test_labels = test["labels"]
    del test

model.evaluate(test_features, test_labels, batch_size=batch_size)

#Save the model
model.save("model_15.h5")

