import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from collections import deque
from global_vars import xm_per_pix, ym_per_pix, image_h
from threshold_utilis import combineThreshold
from perspective_utilis import birdeye
from calibration_utilis import calibrate_camera, undistort

class Line(object):
    def __init__(self, n_last):
        """
        `n_last`: how many last frames will be used to get the average polynomial fits
        """
        self.xpix = None # the pixel position on x axis
        self.ypix = None # the pixel position on y axis

        self.detected = False # indicated whether the lane lines detected

        self.last_fit_px = None # the last polynomial fit coefficients of pixel
        self.last_fit_meter = None # the last polynomial fit coefficents of meter

        self.recent_fits_px = deque(maxlen=n_last) # the last `n_last` polynomial fits coefficients of pixel
        self.recent_fits_meter = deque(maxlen=n_last) # the last `n_last` polynomial fits coefficients of meter

        self.radius = None

    def update(self, new_fit_px, new_fit_meter, detected):
        self.detected = detected
        self.last_fit_px = new_fit_px
        self.last_fit_meter = new_fit_meter
        self.recent_fits_px.append(self.last_fit_px)
        self.recent_fits_meter.append(self.last_fit_meter)
        coeff = np.mean(self.recent_fits_meter, axis=0)
        self.radius = self.get_radius_of_curvature(coeff)

    def get_radius_of_curvature(self, coeff):
        y_val = ym_per_pix * image_h
        a = ((1 + (2 * coeff[0] * y_val + coeff[1]) ** 2) ** 1.5) / np.absolute(2 * coeff[0])
        return a



def fit_by_sliding_window(warp, left_line, right_line, num_wins=9, verbose=False):
    """
    `image`: the warped image
    `return`: the left lane line coordinates and right lane line coordinates
    """
    win_height = np.int(warp.shape[0]/num_wins) # the height of each window
    win_width = 200 # the width of window
    marginx = 100 # the margin of each line
    minpx = 50 # minimum number of pixels in each window
    #get the histgram of the image
    hist = np.sum(warp[np.int(combine_1.shape[0]/2):, :], axis=0)
    midpoint = np.int(warp.shape[1]/2)
    left_base = np.argmax(hist[:midpoint])
    right_base = np.argmax(hist[midpoint:]) + midpoint
    #get the nonzeros index of image
    nonzeros = np.nonzero(warp)
    nonzerox = nonzeros[1]
    nonzeroy = nonzeros[0]
    #set the left and right lane line points 
    left_lane_pts = []
    right_lane_pts = []
    #set the current left_center and right_center point
    cur_left_center = left_base
    cur_right_center = right_base
    out_img = np.dstack((warp, warp, warp))
    #find the window and draw window
    for i in range(num_wins):
        v_low = warp.shape[0]-(i+1)*win_height #the y axis index of window upper horizontal line
        v_high = warp.shape[0] - i*win_height #the y axis index of window lower horizontal line
        h_left_low = cur_left_center - marginx
        h_left_high = cur_left_center + marginx
        h_right_low = cur_right_center - marginx
        h_right_high = cur_right_center + marginx
        cv2.rectangle(out_img, (h_left_low, v_low), (h_left_high, v_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (h_right_low, v_low), (h_right_high, v_high), (0, 255, 0), 2)
        good_left = ((nonzerox >= h_left_low) & (nonzerox < h_left_high) & (nonzeroy >= v_low) & (nonzeroy < v_high)).nonzero()[0]
        good_right = ((nonzerox >= h_right_low) & (nonzerox < h_right_high) & (nonzeroy >= v_low) & (nonzeroy < v_high)).nonzero()[0]
        left_lane_pts.append(good_left)
        right_lane_pts.append(good_right)
        if good_left.shape[0] > minpx:
            cur_left_center = np.int(np.mean(nonzerox[good_left]))
        if good_right.shape[0] > minpx:
            cur_right_center = np.int(np.mean(nonzerox[good_right]))
    left_lane_pts = np.concatenate(left_lane_pts)
    right_lane_pts = np.concatenate(right_lane_pts)
    leftx = nonzerox[left_lane_pts]
    lefty = nonzeroy[left_lane_pts]
    rightx = nonzerox[right_lane_pts]
    righty = nonzeroy[right_lane_pts]
    left_line.xpix = leftx
    left_line.ypix = lefty
    right_line.xpix = rightx
    right_line.ypix = righty
    detected = True
    if not list(left_line.xpix) or not list(left_line.ypix):
        left_fit_px = left_line.last_fit_px
        left_fit_meter = left_line.last_fit_meter
        detected = False
    else:
        left_fit_px = np.polyfit(lefty, leftx, 2)
        left_fit_meter = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)

    if not list(right_line.xpix) or not list(right_line.ypix):
        right_fit_px = right_line.last_fit_px
        right_fit_meter = right_line.last_fit_meter
        detected = False
    else:
        right_fit_px = np.polyfit(righty, rightx, 2)
        right_fit_meter = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_line.update(left_fit_px, left_fit_meter, detected)
    right_line.update(right_fit_px, right_fit_meter, detected)

    #draw the line and polygon on the out_image
    ploty = np.linspace(0, warp.shape[0]-1, warp.shape[0])
    left_fitx = left_fit_px[0]*(ploty**2) + left_fit_px[1]*ploty + left_fit_px[2]
    right_fitx = right_fit_px[0]*(ploty**2) + right_fit_px[1]*ploty + right_fit_px[2]
    left_pair = np.vstack((left_fitx, ploty)).transpose()
    right_pair = np.flipud(np.vstack((right_fitx, ploty)).transpose())
    polyPoints = np.vstack((left_pair, right_pair))
    cv2.polylines(out_img, np.int_([left_pair]), isClosed=False, color=(255, 0, 0), thickness=30)
    cv2.polylines(out_img, np.int_([right_pair]), isClosed=False, color=(0, 0, 255), thickness=30)
    cv2.fillPoly(out_img, np.int_([polyPoints]), color=(100, 255, 50))

    if verbose:
        plt.imshow(out_img)
        plt.show()
    
    return left_line, right_line, out_img

def fit_by_previous_line(warp, left_line, right_line, verbose=False):
    out_img = np.dstack((warp, warp, warp))
    nonzeros = np.nonzero(warp)
    nonzerox = nonzeros[1]
    nonzeroy = nonzeros[0]
    marginx = 100
    left_fit_px = left_line.last_fit_px
    left_fit_meter = left_line.last_fit_meter
    right_fit_px = right_line.last_fit_px
    right_fit_meter = right_line.last_fit_meter
    left_lane_pts = (nonzerox >= (left_fit_px[0]*(nonzeroy**2)+left_fit_px[1]*nonzeroy+left_fit_px[2] - marginx)) & (nonzerox < (left_fit_px[0]*(nonzeroy**2)+left_fit_px[1]*nonzeroy+left_fit_px[2]+marginx))
    right_lane_pts = (nonzerox >= (right_fit_px[0]*(nonzeroy**2)+right_fit_px[1]*nonzeroy+right_fit_px[2] - marginx)) & (nonzerox < (right_fit_px[0]*(nonzeroy**2)+right_fit_px[1]*nonzeroy+right_fit_px[2]+marginx))
    leftx = nonzerox[left_lane_pts]
    lefty = nonzeroy[left_lane_pts]
    rightx = nonzerox[right_lane_pts]
    righty = nonzeroy[right_lane_pts]
    left_line.xpix = leftx
    left_line.ypix = lefty
    right_line.xpix = rightx
    right_line.ypix = righty
    detected = True
    if not list(left_line.xpix) or not list(left_line.ypix):
        left_fit_px = left_line.last_fit_px
        left_fit_meter = left_line.last_fit_meter
        detected = False
    else:
        left_fit_px = np.polyfit(lefty, leftx, 2)
        left_fit_meter = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)

    if not list(right_line.xpix) or not list(right_line.ypix):
        right_fit_px = right_line.last_fit_px
        right_fit_meter = right_line.last_fit_meter
        detected = False
    else:
        right_fit_px = np.polyfit(righty, rightx, 2)
        right_fit_meter = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_line.update(left_fit_px, left_fit_meter, detected)
    right_line.update(right_fit_px, right_fit_meter, detected)

    #draw the line and polygon on the out_image
    ploty = np.linspace(0, warp.shape[0]-1, warp.shape[0])
    left_fitx = left_fit_px[0]*(ploty**2) + left_fit_px[1]*ploty + left_fit_px[2]
    right_fitx = right_fit_px[0]*(ploty**2) + right_fit_px[1]*ploty + right_fit_px[2]
    left_pair = np.vstack((left_fitx, ploty)).transpose()
    right_pair = np.flipud(np.vstack((right_fitx, ploty)).transpose())
    polyPoints = np.vstack((left_pair, right_pair))
    cv2.polylines(out_img, np.int_([left_pair]), isClosed=False, color=(255, 0, 0), thickness=30)
    cv2.polylines(out_img, np.int_([right_pair]), isClosed=False, color=(0, 0, 255), thickness=30)
    cv2.fillPoly(out_img, np.int_([polyPoints]), color=(100, 255, 50))

    if verbose:
        plt.imshow(out_img)
        plt.show()
    
    return left_line, right_line, out_img

def draw_back_to_image(ori_image, lined_image, Minv, verbose=False):
    """
    `ori_image`: original image(distorted)
    `lined_image`: warped image with line drawn
    `Minv`: the inverse transform matrix
    """
    img_size = (ori_image.shape[1], ori_image.shape[0])
    warp_back = cv2.warpPerspective(lined_image, Minv, dsize=img_size, flags=cv2.INTER_LINEAR)
    out_img = cv2.addWeighted(ori_image, 1, warp_back, 0.3, 0)
    if verbose:
        plt.imshow(out_img)
        plt.show()
    return out_img





ret, mtx, dst, rvecs, tvecs = calibrate_camera("camera_cal")
image_files_path = glob.glob("test_images/test*.jpg")
for file in image_files_path:
    image = mpimg.imread(file)
    undistort(image, mtx, dst, verbose=False)
left_line = Line(10)
right_line = Line(10)
file = image_files_path[0]
image = mpimg.imread(file)
und_1 = undistort(image, mtx, dst, verbose=False)
combine_1 = combineThreshold(und_1, 60, 120, 180, 250, verbose=False)
warp, M, Minv = birdeye(combine_1, verbose=True)
_, _, lined_image = fit_by_sliding_window(warp, left_line, right_line, num_wins=9, verbose=False)
draw_back_to_image(image, lined_image, Minv, verbose=True)
for file in image_files_path[1:]:
    image = mpimg.imread(file)
    # gradientThreshold(image, 3, 50, 200, verbose=True)
    und_1 = undistort(image, mtx, dst, verbose=False)
    combine_1 = combineThreshold(und_1, 50, 200, 180, 250)
    warp, M, Minv = birdeye(combine_1, verbose=False)
    _, _, lined_image = fit_by_previous_line(warp, left_line, right_line, verbose=False)
    draw_back_to_image(image, lined_image, Minv, verbose=True)

