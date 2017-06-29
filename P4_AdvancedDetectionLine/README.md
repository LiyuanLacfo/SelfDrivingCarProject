**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `calibration_utilis.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_pts` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_pts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_pts` and `img_pts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
The distorted image:

![distorted image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/distorted_image.jpg)

The undistorted image:

![undistorted image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/undistorted_image.jpg)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

The distorted image:

![distorted image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/distorted_image.jpg)

The undistorted image:

![undistorted image](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/undistorted_image.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 19 through 57 in `threshold_utilis.py`).  

For the gradient threshold, I use both x and y axis sobel operator. And combine them together to get the magnitude of the gradient.

For the color thresholding, I select the saturation channel of HLS image.

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![threshold](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/thresholded_image.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birdeye()`, which appears in lines 9 through 24 in the file `perspective_utilis.py`.  The `birdeye()` function takes as inputs an image (`image`).  I chose the hardcode the source and destination points in the following manner:

```python
h, w = image.shape[0:2]
src = np.array([[0, h-10], [546, 460], [732, 460], [w, h-10]], np.float32)
dst = np.array([[0, h], [0, 0], [w, 0], [w, h]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 710      | 0, 720        | 
| 546, 460      | 0, 0      |
| 732, 460     | 1280, 0      |
| 1280, 710      | 1280, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![before perspective](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/warped_before.jpg)
![after perspective](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/warped_after.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

If there is no frame before, I use ***sliding window*** method to find the lane line points, otherwise I use the fit coefficients from the previous frame to find the lane line points. You can find the steps from line 46 to line 189 in `line_utilis.py`

![sliding window](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/find_lane_line_pts.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature from line 39 to line 42 in `line_utilis.py` and offset from line 57 to line 67 in `main.py`.

For the curvature, I first fit the curve using meter as unit. And then use the [formula of radius of curvature](https://en.wikipedia.org/wiki/Radius_of_curvature) to calculate the number.

For the offset, I first find out the center of image. And then I find the lane center by the lane line points at the bottom of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 191 through 203 in my code in `line_utilis.py` in the function `draw_back_to_image()`.  Here is an example of my result on a test image:

![draw back](https://github.com/LiyuanLacfo/SelfDrivingCarProject/blob/master/P4_AdvancedDetectionLine/example_images/draw_back.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/rftqO17vA5U)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I took the computer vision techniques, such as camera calibration, color and gradient threshold, and perspective transformation. One of the most important step here is the ***threshold***. We need to tune the threshold parameters to make the lane line distinguished.

The method above did not deal with the challenge video well. The failure due to either the very close distance to left border of the lane or the very bright or dark environment. I think we should use more sophisticated method to focus more on the lane line area and filter well despite the bright or dark environment.


