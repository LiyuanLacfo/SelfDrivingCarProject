from os import path
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pickle

def lazy_calibrate_camera(func):
    """
    Decorator for `calibrate_camera` to avoid re-computing camera matrix and distortion coefficients
    """
    calibration_cache = "./camera_cal/calibration_data.pickle"
    def wrapper(*args, **kwargs):
        if path.exists(calibration_cache):
            print("Load the calibration data")
            with open(calibration_cache, "rb") as f:
                data = pickle.load(f)
        else:
            print("Compute the calibration data")
            data = func(*args, **kwargs)
            with open(calibration_cache, "wb") as f:
                pickle.dump(data, f)
            print("Done")
        return data
    return wrapper

@lazy_calibrate_camera
def calibrate_camera(calibration_images_dir, verbose=False):
            """
            `calibration_images_dir`: the directory which contains the calibration chessboard images
            `verbose`: if True, display the chessboard corners
            `return`: calibration coefficients
            """
            assert path.exists(calibration_images_dir), '"{}" must exist and contain calibration images'.format(calibration_images_dir)
            nrow, ncol = 6, 9 # number of corners for each row and column
            files = glob.glob(path.join(calibration_images_dir, "calibration*.jpg"))
            #Create a template chessboard corners object coordinates
            obj_points = np.zeros((6*9, 3), np.float32)
            obj_points[:, 0:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            obj_pts = [] # the object coordinates
            img_pts = [] # the image coordinates

            for file in files:
                image = mpimg.imread(file)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (ncol, nrow), None)
                if ret:
                    img_pts.append(corners)
                    obj_pts.append(obj_points)
                    if verbose:
                        image = cv2.drawChessboardCorners(image, (ncol, nrow), corners, ret)
                        cv2.imshow("image", image)
                        cv2.waitKey(500)
            img_pts = np.array(img_pts)
            obj_pts = np.array(obj_pts)
            #calibrate camera using the chessboard corners
            img_size = (gray.shape[0], gray.shape[1])
            ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
            return ret, mtx, dst, rvecs, tvecs

def undistort(image, mtx, dst, verbose=False):
    und = cv2.undistort(image, mtx, dst)
    if verbose:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(image)
                ax[0].set_title("The distorted image")
                ax[1].imshow(und)
                ax[1].set_title("The undistorted image")
                plt.show()
    return und

ret, mtx, dst, rvecs, tvecs = calibrate_camera("camera_cal")
image_files_path = glob.glob("test_images/test*.jpg")
for file in image_files_path:
    image = mpimg.imread(file)
    undistort(image, mtx, dst, verbose=False)





