import pickle
from sliding_window_utilis import find_car, make_heatmap, heatmap_threshold, draw_labeled_heatmap
import numpy as np
import os
from moviepy.editor import VideoFileClip
import collections

time_window = 15
hot_windows_history = collections.deque(maxlen=time_window)


def pipeline(image):
    windows = []
    for subsample in np.arange(1.3, 2.0, 0.3):
        draw_img, wins = find_car(image, clf=clf, X_Scaler=scaler, y_start=400, y_stop=600, scale=subsample, cspace="RGB")
        windows += wins
    if windows:
        hot_windows_history.append(windows)
    windows = np.concatenate(hot_windows_history)
    draw_image = draw_labeled_heatmap(image, windows, 34)
    return draw_image

if __name__ == "__main__":
    with open("clf_linear_RGB.p", "rb") as f:
        clf = pickle.load(f)["clf"]
    with open("scaler_RGB.p", "rb") as f:
        scaler = pickle.load(f)["scaler"]
    if not os.path.exists("video_output"):
        os.makedirs("video_output")
    video_output = "video_output/project_video_4.mp4"
    clip_1 = VideoFileClip("project_video.mp4")
    frame_clip_1 = clip_1.fl_image(pipeline)
    frame_clip_1.write_videofile(video_output, audio=False)

