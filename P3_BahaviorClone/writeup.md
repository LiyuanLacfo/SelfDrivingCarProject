# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 6 convolutional layers, 6 pooling layers, 6 dropout layers and 4 fully connected layers. The filter size of convolutional layers are all 3x3. The depth are respectively 36, 48, 64, 128, 128 and 128 (model.py lines 153-201)

The model includes ELU layers to introduce nonlinearity(code line 158) and the data is normalized in the model using a Keras lambda layer(code line 150).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 159). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 228). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an RMSprop optimizer, the learning rate is 0.0005 (model.py line 132). I have tried other learning rates such as 0.001, 0.005. But the model can't learn the training data very well, so I reduce the learning rate.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and turning curves especially sharp curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was ***try and error***.

My first step was to use a convolution neural network model similar to the [Nvidia Model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because it contains some convolutional layers which can deal with images input and fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add ***dropout layers***. 

Then I tried to tune the dropout rate for several times.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the sharp curves. To improve the driving behavior in these cases, I collect more data in such areas and retrain the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes. The model consists of 6 convolutional layers, 6 pooling layers, 6 dropout layers and 4 fully connected layers. The filter size of convolutional layers are all 3x3. The depth are respectively 36, 48, 64, 128, 128 and 128

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 Color image   							| 
| Cropping         		| 320x65x3 Color image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 320x65x36 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 160x33x36				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 160x33x48
| ELU					   |										|
| Max pooling  |  2x2 stride, output 80x17x48      									|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 80x17x64
| ELU					   |										|
| Max pooling  |  2x2 stride, output 40x9x64      									|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 40x9x128
| ELU					   |										|
| Max pooling  |  2x2 stride, output 20x5x128      									|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 20x5x128
| ELU					   |										|
| Max pooling  |  2x2 stride, output 10x3x128      							|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x3x128
| ELU					   |										|
| Max pooling  |  2x2 stride, output 5x2x128      							|
| Fully connected		| 1024 neutrons       									|
| Fully connected		| 100 neutrons       									|
| Fully connected		| 50 neutrons       									|
| Fully connected		| 10 neutrons       									|
| Output				| 1 neutrons       									|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Architecture](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/model_arc.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/center_1.jpg)

Here is the image after cropping:
![center lane](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/crop_image.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from the side. These images show what a recovery looks like starting from right to center :

![go back 1](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/go_back_1.jpg)
![go back 2](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/go_back_2.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase the data size in a reasonable way. For example, here is an image that has then been flipped:

![normal](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/image_normal.jpg)
![flip](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P3_BahaviorClone/flip_image.jpg)


After the collection process, I had 67768 number of data points. I then preprocessed this data by crop the top and bottom of the image by 70 and 25 pixels respectively.


I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5-6 as evidenced by students who have already done this project. I used an RMSprop optimizer so that I can adjust learning rate when the model does not learn the training data well.


