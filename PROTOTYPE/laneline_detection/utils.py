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
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
            norm = [(math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2), fit[0], fit[1]), (x1, x2, y1, y2)]
            if norms.keys() is None:
                if (75 > theta) & (theta > 15):
                    norms[theta] = [norm]
            else:
                for t in norms.keys():
                    if abs(t - theta) < 5 & (((float(norms[t][0][0][1]) - fit[0])**2 + (float(norms[t][0][0][2]) - fit[1])**2)**0.5 < 10):
                        norms[t].append(norm)
                        flag = True
                        break
                if flag:
                    flag = False
                    continue
                else:
                    if (75 > theta) & (theta > 15):
                        norms[theta] = [norm]

    return norms


def average_lines(lines, imshape):
    hough_pts = {'m_left': [], 'b_left': [], 'norm_left': [], 'm_right': [], 'b_right': [], 'norm_right': []}
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                m = fit[0]
                b = fit[1]
                print m, b
                norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if m > 0:
                    hough_pts['m_right'].append(m)
                    hough_pts['b_right'].append(b)
                    hough_pts['norm_right'].append(norm)
                if m < 0:
                    hough_pts['m_left'].append(m)
                    hough_pts['b_left'].append(b)
                    hough_pts['norm_left'].append(norm)

    if len(hough_pts['b_left']) != 0 or len(hough_pts['m_left']) != 0 or len(hough_pts['norm_left']) != 0:
        b_avg_left = np.mean(np.array(hough_pts['b_left']))
        m_avg_left = np.mean(np.array(hough_pts['m_left']))
        # print b_avg_left, m_avg_left
        '''y = mx + b'''
        if b_avg_left < imshape[0]:
            xa = 0
            ya = b_avg_left
        else:
            ya = imshape[0]
            xa = (ya - b_avg_left) / m_avg_left
        ya2 = imshape[0] / 3
        xa2 = (ya2 - b_avg_left) / m_avg_left
        left_lane = [int(xa), int(ya), int(xa2), int(ya2)]
    else:
        left_lane = [0, 0, 0, 0]
    if len(hough_pts['b_right']) != 0 or len(hough_pts['m_right']) != 0 or len(hough_pts['norm_right']) != 0:
        b_avg_right = np.mean(np.array(hough_pts['b_right']))
        m_avg_right = np.mean(np.array(hough_pts['m_right']))
        '''y = mx + b'''
        if b_avg_right < imshape[0]:
            xb = 0
            yb = b_avg_right
        else:
            yb = imshape[0]
            xb = (yb - b_avg_right) / m_avg_right
        yb1 = imshape[0] / 3
        xb1 = (yb - b_avg_right) / m_avg_right
        right_lane = [int(xb1), int(yb1), int(xb), int(yb)]
    else:
        right_lane = [0, 0, 0, 0]

    return left_lane, right_lane


def canny_thresh(img, low=30, high=100):
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
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return cv2.addWeighted(gray, 0.8, mask, 1, 0)


def white_enhance(img_rgb):
    lower_white = np.array([140, 140, 140])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img_rgb, lower_white, upper_white)

    return mask


def thresh_white(image):
    lower = np.array([140, 140, 140])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    return mask
