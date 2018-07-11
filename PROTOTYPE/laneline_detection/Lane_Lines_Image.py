import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import utils
import math

imageio.plugins.ffmpeg.download()


# from moviepy.editor import *
# import moviepy.editor
# from IPython.display import HTML

def process_image1():
    # Read in and grayscale the image
    image = cv2.imread('./test_pages/test2.jpg')
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
    vertices2 = np.array([[(imshape[1] / 3, imshape[0]), (imshape[1] / 2, 1.7 * imshape[0] / 3), (2 * imshape[1] / 3, imshape[0])]],
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

    # print(lines)

    # Iterate over the output "lines" and draw lines on a blank image

    norms = utils.hough_filter(lines)

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
    #         if (80 > theta) & (theta > 20):
    #             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    m_avg_right = []
    m_avg_left = []
    b_avg_left = []
    b_avg_right = []
    for t in norms.keys():
        for norm in norms[t]:
            for x1, x2, y1, y2 in norm[1:]:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # Create a "color" binary image to combine with line image

    # color_edges = np.dstack((canny, canny, canny))

    # Draw the lines on the edge image

    image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(image_result)
    plt.show()

    return image_result


def process_image():
    # Read in and grayscale the image
    image = cv2.imread('./test_pages/test2.jpg')
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
    # imshape = image.shape
    # vertices = np.array([[(imshape[1] * 5 / 12, imshape[0] / 2), (imshape[1] * 7 / 12, imshape[0] / 2),
    #                       (imshape[1], imshape[0] * 3 / 4), (imshape[1], imshape[0]),
    #                       (0, imshape[0]), (0, imshape[0] * 3 / 4)]])
    # vertices2 = np.array([[(imshape[1] / 4, imshape[0]), (imshape[1] * 3 / 4, imshape[0]),
    #                        (imshape[1] / 2, imshape[0] / 2)]])
    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    # cv2.fillPoly(mask, vertices2, 0)

    imshape = image.shape
    vertices1 = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                           (2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices1, ignore_mask_color)
    vertices2 = np.array([[(imshape[1] / 3, imshape[0]), (imshape[1] / 2, 1.7 * imshape[0] / 3), (2 * imshape[1] / 3, imshape[0])]],
                         dtype=np.int32)
    cv2.fillPoly(mask, vertices2, 0)
    vertices3 = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                           (0, 7 * imshape[0] / 8)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices3, ignore_mask_color)
    vertices4 = np.array([[(2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0]),
                           (imshape[1], 7 * imshape[0] / 8)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices4, ignore_mask_color)

    masked_edges = cv2.bitwise_and(canny, mask)
    plt.imshow(mask)
    plt.show()
    # plt.imshow(masked_edges)
    # plt.show()

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

    # print(lines)

    # Iterate over the output "lines" and draw lines on a blank image

    norms = utils.hough_filter(lines)

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
    #         if (80 > theta) & (theta > 20):
    #             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    m_avg_right = []
    m_avg_left = []
    b_avg_left = []
    b_avg_right = []
    m_total_right = 0
    n_total_right = 0
    m_total_left = 0
    n_total_left = 0
    b_total_right = 0
    b_total_left = 0
    for t in norms.keys():
        for norm in norms[t]:
            for n, m, b in norm[0:1]:
                if m > 0:
                    m_total_right += m * n
                    n_total_right += n
                    b_total_right += b * n
                else:
                    m_total_left += m * n
                    n_total_left += n
                    b_total_left += b * n
            # for x1, x2, y1, y2 in norm[1:]:
            #     fit = np.polyfit((x1, x2), (y1, y2), 1)
            #     m = fit[0]
            #     b = fit[1]
            #     if m > 0:
            #         m_avg_right.append(m)
            #         b_avg_right.append(b)
            #     else:
            #         m_avg_left.append(m)
            #         b_avg_left.append(b)

    if m_total_left != 0 or b_total_left != 0:
        b_left = b_total_left / n_total_left
        m_left = m_total_left / n_total_left
        # print b_avg_left, m_avg_left
        '''y = mx + b'''
        if b_left < imshape[0]:
            xa = 0
            ya = b_left
        else:
            ya = imshape[0]
            xa = (ya - b_left) / m_left
        ya2 = imshape[0] * 1.8 / 3
        xa2 = (ya2 - b_left) / m_left
        cv2.line(line_image, (int(xa), int(ya)), (int(xa2), int(ya2)), (255, 0, 0), 5)

    if m_total_right != 0 or b_total_right != 0:
        b_right = b_total_right / n_total_right
        m_right = m_total_right / n_total_right
        '''y = mx + b'''
        xtry = imshape[1]
        ytry = imshape[1] * m_right + b_right
        if ytry < imshape[0]:
            xb = xtry
            yb = ytry
        else:
            yb = imshape[0]
            xb = (yb - b_right) / m_right
        yb1 = imshape[0] * 1.8 / 3
        xb1 = (yb1 - b_right) / m_right
        cv2.line(line_image, (int(xb), int(yb)), (int(xb1), int(yb1)), (255, 0, 0), 5)

    plt.imshow(line_image)
    plt.show()
    # Create a "color" binary image to combine with line image

    # color_edges = np.dstack((canny, canny, canny))

    # Draw the lines on the edge image

    image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(image_result)
    plt.show()

    return image_result


process_image1()
process_image()
