## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for HOG feature is between line 24 and 34 of `feature_extraction_utilis.py`   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![vehicle](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/vehicle.jpg)
![non-vehicle](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/non-vehicle.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![hog_feature](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/hog_feature.jpg)

As for the parameters of HOG, the color space I chose was `RGB`. I have tried `YCrCb` and `YUV`, but the performance was bad, since for the test images, I detected a lot of windows bounding non-vehicles. 

For other parameters, `orientations=9`, `cells_per_block=2` and `pixels_per_cell=8`.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and see whether the outline of the car shape is relatively clear. And after feeding the classifier, I also tuned the parameters to increase the validation accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using three kinds of features which are HOG feature, color feature and spatial feature.

The code is between line 43 and 55 of `classifier_utilis.py`

First I extract each of the 3 features above. Second I concatenate all the features to be a vector. Third I feed this concatenated feature into the SVM classifier. 

I choose to use linear SVM since it is faster when we search the object in the image later.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is between line 7 and 81 of `sliding_window_utilis.py`. 

To increase the speed of search, first I get the HOG feature of the whole image. 

Second, instead of setting the overlapping window, I set the `cell_per_step` to control the move of window in unit of cell. I use `cell_per_step = 2` which means the window slides by 2 cells per step. 

Third, the window size is `64x64`. Here I chose that size because the training image is of size `64x64`, and it is convenient to use that size when get the local image HOG feature from the whole image HOG feature.

Fourth, I use 3 different window scales which are 1.3, 1.6 and 1.9. For the smaller vehicles that are far away from the car with camera, we need to keep their size as much as possible, that is why I use 1.3. And for the larger cars, it is good to scale down them to make a good classification. Also the usage of 3 scales makes more boxes for the vehicles and distinguishes cars from non-cars. 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used `RGB` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I have tried several times to choose number of bins for color histogram and the spatial binning size to optimize the classifier.

Here are some example images:

![test image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/test1.jpg)
![test image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/test2.jpg)
![test image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/test3.jpg)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/iB_L-Z0It6c)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

To filter false positives, I used 2 methods. The first one is threshold, and the second one is to consider a bunch of consecutive frames, and use these frames to make the boxes. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:



### Here are six frames and their corresponding heatmaps:

![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame20.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame21.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame37.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame38.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame39.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/heatmap_frame40.jpeg)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame20.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame21.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame37.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame38.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame39.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/labeled_array_frame40.jpeg)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame20.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame21.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame37.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame38.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame39.jpeg)
![alt text](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P5_Vehicle_Detection/example_images/draw_img_frame40.jpeg)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The drawback of the combination of SVM and HOG is easy to understand. There are a lot of parameters needed to be tuned, such as the HOG number of orientations, the number of bins of color histogram. The parameters chosen for this project may not fit another environment. So I will try to use deep learning method such as CNN to do feature extraction. I think it would be more robust.


