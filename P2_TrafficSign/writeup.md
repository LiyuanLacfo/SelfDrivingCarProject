# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32$\times$32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I have tried a few training tests, the performance of training on gray images are better than on color images.

Here is an example of a traffic sign image before and after grayscaling.

![Before grayscaling](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/color_image.jpg)


![After grayscaling](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/gray_image.jpg)

As a last step, I normalized the image data because we want to avoid numerical instability.

I decided to generate additional data because the distribution of classes is far from uniform. Some classes have much fewer data than others.

To add more data to the the data set, I use the following 4 techniques, ***translation***, ***rotation***, ***scaling*** and ***shearing***. Because we just change the position or the angle of the image shape. For example, the translation of traffic sign position in the image should not affect the classification result.

Here is an example of an original image and an augmented image:

![original image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/before_trans.jpg)

![augmented image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/after_trans.jpg)


The difference between the original data set and the augmented data set is the position of the traffic sign in the image is translated from the original image.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x32
| Max pooling  |  2x2 stride, output 8x8x32      									|
| Fully connected		| 512 neutrons       									|
| Softmax				| 43 neutrons       									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I use *adam optimizer* to train my network. The batch size is 128 and epoch number is 100. The learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of  98.3%
* test set accuracy of 98.03%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x108				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 28x28x108
| Max pooling  |  1x1 stride, output 28x28x108      									|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 26x26x64
| Max pooling  |  1x1 stride, output 26x26x64      									|
| Fully connected		| 5000 neutrons       									|
| Fully connected		| 3000 neutrons       									|
| Fully connected		| 1000 neutrons       									|
| Softmax				| 43 neutrons       							|

* What were some problems with the initial architecture?

The accuracy on the validation set is only 84%.
The accuracy on the test set is very high, thus the problem here is overfitting. We should reduce the size of neural network.

* How was the architecture adjusted and why was it adjusted? 

Since the model above is overfitting. I try to reduce the size of network and use dropout.

(1)I reduce the number of fully connected layers such that there is only 2 fully connected layers.

(2)And I turn the color images into grayscale images. 

(3) I use dropout to the training set.

(4) I use *SAME PADDING* instead of *VALID PADDING*


* Which parameters were tuned? How were they adjusted and why?
(1) The number of fully connected layers. Since I need to reduce the size of network architecture size. 
(2) The number of neutrons in the fully connected layer. The reason is the same as above.
(3) The number of output of the convolutional layers. Also I want to reduce the size of network. And I changed the *SAME PADDING* into *VALID PADDING*.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

(1)Dropout is very important in this problem. Without dropout, the training accuracy would soon go up to 99%, but the validation accuracy is just about 90%. Because without dropout the training process would learn a lot noise in the training set.

(2) Learning rate should not be large. In fact large learning rate does not mean fast learning. Sometimes, large learning rate means very slow learning. It would overshoot if the learning rate is high.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/figure_0.jpg) 

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/figure_1.jpg) 

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/figure_2.jpg)

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/figure_3.jpg) 

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P2_TrafficSign/figure_4.jpg)

The quality of these images are pretty well. So it is not difficult for our model to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      | Vehicles over 3.5 metric tons prohibited  									| 
| Speed limit (20km/h)     	| Speed limit (20km/h)										|
| Keep right					| Keep right											|
| Turn right ahead	      	| Turn right ahead					 				|
| Right-of-way at the next intersection | Right-of-way at the next intersection      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100% which is greater than on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		| Vehicles over 3.5 metric tons prohibited    									| 
| 1.0     				| 	Speed limit (20km/h)								|
| 0.994				| 		Keep right								|
| 0.995      			| 	Turn right ahead				 				|
| 0.999			    |   Right-of-way at the next intersection   							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


