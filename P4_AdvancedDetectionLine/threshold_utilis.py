import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def clahe(image, contrast_limit=2.0, gridSize=(8, 8), verbose=False):
    gray = grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=gridSize)
    cl1 = clahe.apply(gray)
    if verbose:
        cv2.imshow("clahe", cl1)
        cv2.waitKey(500)
    return cl1

def gradientThreshold(image, ksize, low_threshold, high_threshold, verbose=False):
    gray = clahe(image, 2.0, (16, 16))
    gray = grayscale(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.sqrt(sobelx**2+sobely**2)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    out_img = np.zeros_like(gray)
    out_img[(scaled_sobelx>=low_threshold) & (scaled_sobelx<=high_threshold)] = 1
    if verbose:
        plt.imshow(out_img, cmap="gray")
        plt.title("Gradient Threshold")
        plt.show()
    return out_img

def colorThreshold(image, low_threshold, high_threshold, verbose=False):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_image = hls[:, :, 2]
    out_img = np.zeros_like(s_image)
    out_img[(s_image>=low_threshold) & (s_image <= high_threshold)] = 1
    if verbose:
        plt.imshow(out_img, cmap="gray")
        plt.title("Color Threshold")
        plt.show()
    return out_img

def combineThreshold(image, gThresholdLow, gThresholdHigh, cThresholdLow, cThresholdHigh, verbose=False, ksize=3):
    g_image = gradientThreshold(image, ksize, gThresholdLow, gThresholdHigh)
    c_image = colorThreshold(image, cThresholdLow, cThresholdHigh)
    out_img = np.zeros_like(g_image)
    out_img[(g_image==1) | (c_image==1)] = 255
    kernel = np.ones((5, 5), np.uint8)
    close = cv2.morphologyEx(out_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if verbose:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(out_img, cmap="gray")
        ax[1].imshow(close, cmap="gray")
        plt.show()
    return out_img



image_files_path = glob.glob("test_images/test*.jpg")
for file in image_files_path:
    image = mpimg.imread(file)
    # gradientThreshold(image, 3, 50, 200, verbose=True)
    combine_1 = combineThreshold(image, 50, 200, 200, 250, verbose=False)
