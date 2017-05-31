# Finding Lane Lines on the Road
### Reflection
###1. Briefly describe my pipeline for detecting the lane line in the image.
My pipeline consists of 6 steps. 

- The first step is to convert the RGB image into grayscale image. 
- And the second step is add a gaussian blur to the grayscale image. We can reduce the impact of noise when detecting the edge. 
- The third step is to use Canny edge detection algorithm on the blur added grayscale image. 
- The forth step is to use a mask to leave only a selected region of interest of image after using edge detection algorithm. And the other else region is set to be black. 
- The fifth step is to use Hough transform algorithm to draw lines on the image which is the output of the previous step. 
- And the final step is to combine the image which has been drawn line and the original image.

To draw just two lines of the lane lines, we need to modify the `draw_line` function. I create a class called `Line`. It contains `x1, y1, x2, y2` which are 2 points for representing one line. Also it has slope `slope` and intercept `bias` of the line.

After hough transform, we can get several lines. I divide these lines by the slope into two groups. One group contains the lines with positive slope and the other consists of lines which have negative slopes. Then I calculate the mean of slope for each group. And use these two averages of line slopes to represent the lane lines slope respectively. 

Here is an output image of the pipeline,
![Image of lane lines](https://github.com/LiyuanLacfo/Udacity_SelfDrivingCar_NanoDeg/blob/master/add_line_test_images/addLine_solidWhiteCurve.jpg)


### 2.The potential shortcoming of my pipeline
One potential shortcoming is that if there are trees along the road. The we would detect the tree as well.

Another shortcoming is that when we meet shadow, the lane line would not be very clear to be detected.

### 3. Possible improvement to the pipe line
For the first shortcoming, we could select line to be drawn by the slope. Since the slope of tree lines would always be near 0.

And for the second shortcoming, I have not come up with a good idea. Maybe we could use more sophisticated algorithm.







