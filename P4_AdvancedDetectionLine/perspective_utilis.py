import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from threshold_utilis import combineThreshold

def birdeye(image, verbose=False):
    h, w = image.shape[0:2]
    src = np.array([[0, h-10], [546, 460], [732, 460], [w, h-10]], np.float32)
    dst = np.array([[0, h], [0, 0], [w, 0], [w, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (w, h)
    warp = cv2.warpPerspective(image, M, dsize=img_size, flags=cv2.INTER_LINEAR)
    if verbose:
        fig, ax = plt.subplots()
        # ax[0].set_title("Before Transform")
        # ax[0].imshow(image, cmap="gray")
        ax.set_title("After Transform")
        ax.imshow(warp, cmap="gray")
        plt.show()
    return warp, M, Minv

# image_files_path = glob.glob("test_images/test*.jpg")
# for file in image_files_path:
#     image = mpimg.imread(file)
#     # gradientThreshold(image, 3, 50, 200, verbose=True)
#     combine_1 = combineThreshold(image, 50, 200, 200, 250, verbose=False)
#     warp, _, _ = birdeye(combine_1, verbose=True)

