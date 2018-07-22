import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import utils
import Framework
import math

imageio.plugins.ffmpeg.download()


def process_image():
    image = cv2.imread('./test_pages/test10.jpg')
    print image.size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enhance_white = utils.white_enhance(image)
    enhance_yellow = utils.yellow_enhance(image)
    final_enhance = cv2.addWeighted(enhance_white, 0.8, enhance_yellow, 1, 0)
    # rt, bianry = cv2.threshold(final_enhance, 127, 255, cv2.THRESH_BINARY)
    # t, binary = cv2.threshold(final_enhance, 0, 255, cv2.THRESH_OTSU, cv2.THRESH_BINARY)
    # plt.imshow(binary)
    # plt.show()
    equ = cv2.equalizeHist(final_enhance, plt.gray())
    plt.imshow(equ)
    plt.show()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(final_enhance, (kernel_size, kernel_size), 0)

    # plt.imshow(blur_gray, plt.gray())
    # plt.show()

    # Define our parameters for Canny and apply
    canny = utils.canny_thresh(blur_gray)
    hls = utils.hls_thresh(image)
    white = utils.thresh_white(image)
    yellow = utils.thresh_yellow(image)

    # cv2.imwrite("canny.png", canny)
    plt.imshow(canny, plt.gray())
    plt.show()

    combined = np.zeros_like(canny)
    combined[(canny > 0) | (hls > 0) | (white > 0) | (yellow > 0)] = 1

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
    vertices_least = np.array([[(imshape[1] / 9, imshape[0]), (imshape[1] * 1.4 / 3, 1.7 * imshape[0] / 3),
                                (1.6 * imshape[1] / 3, 1.7 * imshape[0] / 3), (8 * imshape[1] / 9, imshape[0])]],
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

    plt.imshow(masked_edges_base)
    plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 35  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 45  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines_whole on

    # Run Hough on edge detected image
    # Output "lines_whole" is an array containing endpoints of detected line segments
    lines_whole = cv2.HoughLinesP(masked_edges_whole, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)
    lines_base = cv2.HoughLinesP(masked_edges_base, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_least = cv2.HoughLinesP(masked_edges_least, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)
    lines_left = cv2.HoughLinesP(masked_edges_left, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_right = cv2.HoughLinesP(masked_edges_right, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

    # norms = utils.hough_filter(lines_least)
    # for t in norms.keys():
    #     for norm in norms[t]:
    #         for x1, x2, y1, y2 in norm[1:]:
    #             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    l, r = utils.detect_lines(lines_least, imshape)
    if l and r:
        line_image = utils.draw_lines(lines_least, line_image, imshape)
        line_result = lines_least
    elif (not l) and r:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
        else:
            line_image = utils.draw_lines(lines_left, line_image, imshape)
            line_result = lines_left
    elif (not r) and l:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
        else:
            line_image = utils.draw_lines(lines_right, line_image, imshape)
            line_result = lines_right
    else:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
        else:
            line_image = utils.draw_lines(lines_whole, line_image, imshape)
            line_result = lines_whole

    # Iterate over the output "lines_whole" and draw lines_whole on a blank image
    if line_image is not None:
        image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        image_result = image

    Framework.is_courtesy(image_result, line_result, [(0, 100), (100, 400)])

    image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

    plt.imshow(image_result)
    plt.show()

    cv2.imwrite("imageresult.png", image_result)

    return image_result


process_image()
