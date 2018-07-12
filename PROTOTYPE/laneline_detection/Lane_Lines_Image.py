import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import utils
import math

imageio.plugins.ffmpeg.download()


def process_image1():
    image = cv2.imread('./test_pages/test_curb.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enhance_white = utils.white_enhance(image)
    enhance_yellow = utils.yellow_enhance(image)
    final_enhance = cv2.addWeighted(enhance_white, 0.8, enhance_yellow, 1, 0)
    plt.imshow(final_enhance)
    plt.show()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(final_enhance, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    canny = utils.canny_thresh(blur_gray)
    hls = utils.hls_thresh(image)
    white = utils.thresh_white(image)
    yellow = utils.thresh_yellow(image)

    combined = np.zeros_like(canny)
    combined[(canny > 0) | (hls > 0) | (white > 0) | (yellow > 0)] = 1

    plt.imshow(canny)
    plt.show()

    # Next we'll create a masked canny image using cv2.fillPoly()
    mask = np.zeros_like(canny)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices1 = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                           (2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices1, ignore_mask_color)
    vertices2 = np.array(
        [[(imshape[1] / 3, imshape[0]), (imshape[1] / 2, 1.7 * imshape[0] / 3), (2 * imshape[1] / 3, imshape[0])]],
        dtype=np.int32)
    cv2.fillPoly(mask, vertices2, 0)
    vertices3 = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                           (0, 7 * imshape[0] / 8)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices3, ignore_mask_color)
    vertices4 = np.array([[(2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0]),
                           (imshape[1], 7 * imshape[0] / 8)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices4, ignore_mask_color)
    masked_edges = cv2.bitwise_and(canny, mask)
    plt.imshow(masked_edges)
    plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 35  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    norms = utils.hough_filter(lines)

    for t in norms.keys():
        for norm in norms[t]:
            for x1, x2, y1, y2 in norm[1:]:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(image_result)
    plt.show()

    return image_result


def process_image():
    # Read in and grayscale the image
    image = cv2.imread('./test_pages/test_curb.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enhance_white = utils.white_enhance(image)
    enhance_yellow = utils.yellow_enhance(image)
    final_enhance = cv2.addWeighted(enhance_white, 0.8, enhance_yellow, 1, 0)
    rt, binary = cv2.threshold(final_enhance, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY)
    plt.imshow(binary)
    plt.show()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    canny = utils.canny_thresh(blur_gray)
    hls = utils.hls_thresh(image)
    white = utils.thresh_white(image)
    yellow = utils.thresh_yellow(image)

    combined = np.zeros_like(canny)
    combined[(canny > 0) | (hls > 0) | (white > 0) | (yellow > 0)] = 1

    plt.imshow(combined)
    plt.show()

    mask_whole = np.zeros_like(canny)
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
    cv2.fillPoly(mask_least, vertices_least, ignore_mask_color)
    cv2.fillPoly(mask_least, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_left, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_left, vertices_left_triangle, ignore_mask_color)
    cv2.fillPoly(mask_left, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_right, vertices_base_roi, ignore_mask_color)
    cv2.fillPoly(mask_right, vertices_middle_triangle, 0)
    cv2.fillPoly(mask_right, vertices_right_triangle, ignore_mask_color)

    masked_edges_whole = cv2.bitwise_and(canny, mask_whole)
    masked_edges_least = cv2.bitwise_and(canny, mask_least)
    masked_edges_left = cv2.bitwise_and(canny, mask_left)
    masked_edges_right = cv2.bitwise_and(canny, mask_right)
    plt.imshow(masked_edges_least)
    plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 35  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines_whole on

    # Run Hough on edge detected image
    # Output "lines_whole" is an array containing endpoints of detected line segments
    lines_whole = cv2.HoughLinesP(masked_edges_whole, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)
    lines_least = cv2.HoughLinesP(masked_edges_least, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)
    lines_left = cv2.HoughLinesP(masked_edges_left, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_right = cv2.HoughLinesP(masked_edges_right, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

    l, r = utils.detect_lines(lines_least, imshape)
    if l & r:
        line_image = utils.draw_lines(lines_least, line_image, imshape)
    elif not l & r:
        line_image = utils.draw_lines(lines_left, line_image, imshape)
    elif not r & l:
        line_image = utils.draw_lines(lines_right, line_image, imshape)
    else:
        line_image = utils.draw_lines(lines_whole, line_image, imshape)

    # print(lines_whole)

    # Iterate over the output "lines_whole" and draw lines_whole on a blank image
    image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(image_result)
    plt.show()

    return image_result


process_image()
