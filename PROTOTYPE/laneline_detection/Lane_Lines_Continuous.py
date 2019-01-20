import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
import utils


def get_line_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    correct = utils.preprocess(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(correct, (kernel_size, kernel_size), 0)
    # bilateral = cv2.bilateralFilter(image, -1, 0.3, 10)

    # Define our parameters for Canny and apply
    canny = utils.canny_thresh(blur_gray)

    mask_whole = np.zeros_like(canny)
    mask_base = np.zeros_like(canny)
    mask_least = np.zeros_like(canny)
    mask_left = np.zeros_like(canny)
    mask_right = np.zeros_like(canny)
    ignore_mask_color = 255

    imshape = image.shape
    vertices_base_roi = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                                   (2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0])]],
                                 dtype=np.int32)
    vertices_least = np.array([[(imshape[1] / 9, imshape[0]), (imshape[1] * 1.2 / 3, 1.7 * imshape[0] / 3),
                                (1.8 * imshape[1] / 3, 1.7 * imshape[0] / 3), (8 * imshape[1] / 9, imshape[0])]],
                              dtype=np.int32)
    vertices_middle_triangle = np.array(
        [[(imshape[1] / 3, imshape[0]), (imshape[1] / 2, 1.7 * imshape[0] / 3), (2 * imshape[1] / 3, imshape[0])]],
        dtype=np.int32)
    vertices_left_triangle = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                                        (0, 7 * imshape[0] / 8)]], dtype=np.int32)
    vertices_right_triangle = np.array([[(2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0]),
                                         (imshape[1], 7 * imshape[0] / 8)]], dtype=np.int32)

    cv2.fillPoly(mask_whole, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_whole, vertices_right_triangle, ignore_mask_color)
    cv2.fillPoly(mask_whole, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_whole, vertices_left_triangle, ignore_mask_color)
    cv2.fillPoly(mask_base, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_base, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_least, vertices_least, ignore_mask_color)
    cv2.fillPoly(mask_least, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_left, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_left, vertices_left_triangle, ignore_mask_color)
    cv2.fillPoly(mask_left, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_right, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_right, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_right, vertices_right_triangle, ignore_mask_color)

    masked_edges_whole = cv2.bitwise_and(canny, mask_whole)
    masked_edges_base = cv2.bitwise_and(canny, mask_base)
    masked_edges_least = cv2.bitwise_and(canny, mask_least)
    masked_edges_left = cv2.bitwise_and(canny, mask_left)
    masked_edges_right = cv2.bitwise_and(canny, mask_right)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 45  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines_whole on

    # Run Hough on edge detected image
    # Output "lines_whole" is an array containing endpoints of detected line segments
    lines_whole = cv2.HoughLinesP(masked_edges_whole, rho, theta, threshold, np.array([]),
                                  30, max_line_gap)
    lines_base = cv2.HoughLinesP(masked_edges_base, 1, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    # lines_least = cv2.HoughLinesP(masked_edges_least, 1, theta, 35, np.array([]),
    #                               40, max_line_gap)
    lines_left = cv2.HoughLinesP(masked_edges_left, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_right = cv2.HoughLinesP(masked_edges_right, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

    l, r = utils.detect_lines(lines_base, imshape)
    if l and r:
        line_image, ml, bl, mr, br = utils.draw_lines_continuous(lines_base, line_image, imshape)
    elif (not l) and r:
        line_image, ml, bl, mr, br = utils.draw_lines_continuous(lines_left, line_image, imshape)
    elif (not r) and l:
        line_image, ml, bl, mr, br = utils.draw_lines_continuous(lines_right, line_image, imshape)
    else:
        line_image, ml, bl, mr, br = utils.draw_lines_continuous(lines_whole, line_image, imshape)

    return imshape, ml, bl, mr, br


def process_image(image, line_image):
    # Iterate over the output "lines_whole" and draw lines_whole on a blank image
    if line_image is not None:
        image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        image_result = image

    # Framework.is_courtesy(image_result, line_result, [(0, 100), (100, 400)])

    return image_result


# video = cv2.VideoCapture("test7.mp4")
# success, frame = video.read()
# if success:
#     line_image = np.copy(frame) * 0
#
# writer = cv2.VideoWriter('./lalala2.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (1280, 720))
# # last time's m and b, also the experienced answer
# lml = lbl = lmr = lbr = 0
# # last time's error of m and b
# xml = xbl = xmr = xbr = 5
# # this time's m and b
# imshape, ml, bl, mr, br = get_line_image(frame)
# errorness1 = 105
# errorness2 = 5
# line_image = utils.draw_lines_mb(ml, bl, mr, br, line_image, imshape)
#
# # Using Kalman Filter to get a better result
# while success:
#     frame = process_image(frame, line_image)
#     writer.write(frame)
#
#     # cv2.imshow("window", frame)
#     # cv2.waitKey(0)
#     success, frame = video.read()
#     lml = ml
#     lbl = bl
#     lmr = mr
#     lbr = br
#     imshape, ml, bl, mr, br = get_line_image(frame)
#
#     if ml == 0:
#         ml = lml
#         bl = lbl
#     if mr == 0:
#         mr = lmr
#         br = lbr
#
#     Kg2ml = (errorness2**2 + xml**2) / (errorness2**2 + xml**2 + errorness1**2)
#     ml = lml + (ml - lml) * Kg2ml**0.5
#     xml = 5 * (1 - Kg2ml)**0.5
#
#     Kg2bl = (errorness2 ** 2 + xbl ** 2) / (errorness2 ** 2 + xbl ** 2 + errorness1 ** 2)
#     bl = lbl + (bl - lbl) * Kg2bl ** 0.5
#     xbl = 5 * (1 - Kg2bl) ** 0.5
#
#     Kg2mr = (errorness2 ** 2 + xmr ** 2) / (errorness2 ** 2 + xmr ** 2 + errorness1 ** 2)
#     mr = lmr + (mr - lmr) * Kg2mr ** 0.5
#     xmr = 5 * (1 - Kg2mr) ** 0.5
#
#     Kg2br = (errorness2 ** 2 + xbr ** 2) / (errorness2 ** 2 + xbr ** 2 + errorness1 ** 2)
#     br = lbr + (br - lbr) * Kg2br ** 0.5
#     xbr = 5 * (1 - Kg2br) ** 0.5
#
#     # print frame
#     line_image = np.copy(frame) * 0
#     line_image = utils.draw_lines_mb(ml, bl, mr, br, line_image, imshape)
