import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def hough_filter(lines):
    norms = {}
    flag = False
    if lines is None:
        return norms
    for line in lines:
        for x1, y1, x2, y2 in line:
            theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
            norm = [math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2),(x1, x2, y1, y2)]
            if norms.keys() is None:
                if (80 > theta) & (theta > 20):
                    norms[theta] = [norm]
            else:
                for t in norms.keys():
                    if abs(t - theta) < 5:
                        norms[t].append(norm)
                        flag = True
                        break
                if flag:
                    flag = False
                    continue
                else:
                    if (80 > theta) & (theta > 20):
                        norms[theta] = [norm]

    return norms


def canny_thresh(img, low=50, high=150):
    canny = cv2.Canny(img, low, high)
    binary_out = np.zeros_like(canny)
    binary_out[canny == 255] = 1
    return binary_out


def hls_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, 2]
    binary_out = np.zeros_like(channel)
    binary_out[(channel > 150) & (channel <= 255)] = 1

    return binary_out


def luv_thresh(img, thresh=(160, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1

    return binary_output


def thresh_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([18, 60, 46])
    upper = np.array([34, 180, 250])
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def yellow_enhance(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([15, 80, 46])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    plt.imshow(mask, plt.gray())
    plt.show()
    gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

    return cv2.addWeighted(gray, 0.8, mask, 1, 0)


def white_enhance(img_rgb):
    lower_white = np.array([140, 140, 140])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img_rgb, lower_white, upper_white)
    plt.imshow(mask, plt.gray())
    plt.show()

    return mask


def thresh_white(image):
    lower = np.array([140, 140, 140])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    return mask
