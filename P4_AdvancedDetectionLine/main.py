import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob 
from calibration_utilis import calibrate_camera, undistort
from threshold_utilis import combineThreshold
from line_utilis import Line, fit_by_sliding_window, fit_by_previous_line, draw_back_to_image
from perspective_utilis import birdeye
from global_vars import xm_per_pix, ym_per_pix
import os
from moviepy.editor import VideoFileClip

frame_cnt = 0 #the number of frames processed
left_line = Line(10) # the left lane line 
right_line = Line(10) # the right lane line

def get_final_output(image, birdeye_img, binary_img, radius, offset):
    """
    `image`: the orginal image with line added
    `birdeye_img`: birdeye's image
    `binary_img`: binary image after threshold
    `left_line`: detected left line
    `right_line`: detected right line
    `radius`: radius of curvature
    `offset`: the offset to the center of the road
    `return`: the image added radius of curvature, offset, and birdeye image and binary_img
    """
    h, w, _ = image.shape
    thumb_ratio = 0.2
    thumb_w = int(w * thumb_ratio)
    thumb_h = int(h * thumb_ratio)
    off_x = 20
    off_y = 15
    #add a background at the upper part of the image
    mask = image.copy()
    cv2.rectangle(mask, (0, 0), (w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    image = cv2.addWeighted(mask, 0.2, image, 0.8, 0)

    #add binary image
    thumb_binary = cv2.resize(binary_img, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack((thumb_binary, thumb_binary, thumb_binary))
    image[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    #add bird eye image
    thumb_birdeye = cv2.resize(birdeye_img, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack((thumb_birdeye, thumb_birdeye, thumb_birdeye))
    image[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*off_x+2*thumb_w, :] = thumb_birdeye

    #add radius curvature and offset to lane center
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "curvature radius: {:.02f}m".format(radius), (820, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "offset to road center: {:.02f}m".format(offset), (820, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

def compute_offset(left_line, right_line, image_width):
    if left_line.detected and right_line.detected:
        left_line_bottom = np.mean(left_line.xpix[left_line.ypix > 0.9*left_line.ypix.max()])
        right_line_bottom = np.mean(right_line.xpix[right_line.ypix > 0.9*right_line.ypix.max()])
        image_midpoint = image_width / 2
        offset = (right_line_bottom + left_line_bottom)/2 - image_midpoint
        # offset = (left_line.last_fit_px[2]+right_line.last_fit_px[2])/2 - image_midpoint
        offset_m = offset * xm_per_pix
    else:
        offset_m = -1
    return offset_m

def pipeline(image, verbose=False):
    global left_line, right_line, frame_cnt
    undistort_img = undistort(image, mtx, dist)
    binary_img = combineThreshold(undistort_img, 60, 120, 180, 250, verbose=False)
    birdeye_img, M, Minv = birdeye(binary_img, verbose=False)
    if frame_cnt == 0:
        left_line, right_line, lined_image = fit_by_sliding_window(birdeye_img, left_line, right_line, num_wins=9, verbose=False)
    else:
        left_line, right_line, lined_image = fit_by_previous_line(birdeye_img, left_line, right_line, verbose=False)
    blend_img = draw_back_to_image(image, lined_image, Minv, verbose=False)
    left_radius, right_radius = left_line.radius, right_line.radius
    radius = (left_radius+right_radius)/2
    # radius = left_radius
    offset = compute_offset(left_line, right_line, image.shape[1])
    final_img = get_final_output(blend_img, birdeye_img, binary_img, radius, offset)
    frame_cnt += 1
    if verbose:
        plt.imshow(final_img)
        plt.show()
    return final_img


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    # test_path = "test_images"
    # for file in os.listdir(test_path):
    #     image = mpimg.imread(os.path.join(test_path, file))
    #     out_img = pipeline(image, verbose=True)
    #     cv2.imwrite("output_images/{}".format(file), out_img)
    # if not os.path.exists("test_videos_output"):
    #     os.makedirs("test_videos_output")
    # video_output = 'test_videos_output/output.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # frame_clip = clip1.fl_image(pipeline)
    # frame_clip.write_videofile(video_output, audio=False)
    # if not os.path.exists("challenge_video_output"):
    #     os.makedirs("challenge_video_output")
    # challenge_output = "challenge_video_output/challenge.mp4"
    # clip2 = VideoFileClip("challenge_video.mp4")
    # frame_clip_2 = clip2.fl_image(pipeline)
    # frame_clip_2.write_videofile(challenge_output, audio=False)

    if not os.path.exists("harder_challenge_video_output"):
        os.makedirs("harder_challenge_video_output")
    harder_challenge_output = "harder_challenge_video_output/challenge.mp4"
    clip3 = VideoFileClip("harder_challenge_video.mp4")
    frame_clip_3 = clip3.fl_image(pipeline)
    frame_clip_3.write_videofile(harder_challenge_output, audio=False)







