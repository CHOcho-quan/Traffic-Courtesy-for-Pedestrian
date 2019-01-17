import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import utils
import math
import glob
imageio.plugins.ffmpeg.download()


def process_image(path, out):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # mag = utils.mag_thresh(gray)
    # dir = utils.dir_threshold(gray)
    # abs = utils.abs_sobel_thresh(gray)
    correct = utils.preprocess(image)
    #plt.imshow(correct, plt.gray())
    #plt.show()
    # plt.imshow(dir, plt.gray())
    # plt.show()

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    plt.imshow(dst, plt.gray())
    plt.show()

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(correct, (kernel_size, kernel_size), 0)
    # bilateral = cv2.bilateralFilter(image, -1, 0.3, 10)

    # Define our parameters for Canny and apply
    canny = utils.canny_thresh(blur_gray)
    # canny2 = utils.canny_thresh(bilateral)

    # cv2.imwrite("canny.png", canny)
    # plt.subplot(211)
    #plt.imshow(canny, plt.gray())
    # plt.subplot(212)
    # plt.imshow(canny2)
    #plt.show()

    mask_whole = np.zeros_like(canny)
    mask_base = np.zeros_like(canny)
    mask_least = np.zeros_like(canny)
    mask_left = np.zeros_like(canny)
    mask_right = np.zeros_like(canny)
    ignore_mask_color = 255

    '''
    Using Multi-masks to Define ROI
    '''
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

    #plt.subplot(211)
    #    plt.imshow(masked_edges_least)
    #   plt.subplot(212)
    #   plt.imshow(masked_edges_base)
    #   plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 35  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines_whole on

    # Run Hough on edge detected image
    # Output "lines_whole" is an array containing endpoints of detected line segments
    lines_whole = cv2.HoughLinesP(masked_edges_whole, rho, theta, threshold, np.array([]),
                                  30, max_line_gap)
    lines_base = cv2.HoughLinesP(masked_edges_base, 1, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_least = cv2.HoughLinesP(masked_edges_least, 1, theta, 35, np.array([]),
                                  40, max_line_gap)
    lines_left = cv2.HoughLinesP(masked_edges_left, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
    lines_right = cv2.HoughLinesP(masked_edges_right, rho, theta, threshold, np.array([]),
                                  min_line_length, max_line_gap)

    # norms = utils.hough_filter(lines_least)
    # for t in norms.keys():
    #     for norm in norms[t]:
    #         for x1, x2, y1, y2 in norm[1:]:
    #             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    l, r = utils.detect_lines(lines_base, imshape)
    if l and r:
        line_image = utils.draw_lines(lines_base, line_image, imshape)
        line_result = lines_least
        mask_result = mask_base
    elif (not l) and r:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
            mask_result = mask_base
        else:
            line_image = utils.draw_lines(lines_left, line_image, imshape)
            line_result = lines_left
            mask_result = mask_left
    elif (not r) and l:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
            mask_result = mask_base
        else:
            line_image = utils.draw_lines(lines_right, line_image, imshape)
            line_result = lines_right
            mask_result = mask_right
    else:
        # using base mask to try if OK
        l, r = utils.detect_lines(lines_base, imshape)
        if l & r:
            line_image = utils.draw_lines(lines_base, line_image, imshape)
            line_result = lines_base
            mask_result = mask_base
        else:
            line_image = utils.draw_lines(lines_whole, line_image, imshape)
            line_result = lines_whole
            mask_result = mask_whole

    # Iterate over the output "lines_whole" and draw lines_whole on a blank image
    if line_image is not None:
        image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        image_result = image

    # Framework.is_courtesy(image_result, line_result, [(0, 100), (100, 400)])
    '''
    plt.subplot(221)
    plt.imshow(mask_result)
    plt.subplot(222)
    plt.imshow(image_result)
    plt.subplot(223)
    plt.imshow(masked_edges_base)
    plt.subplot(224)
    plt.imshow(masked_edges_base)
    plt.show()

    plt.imshow(image_result)
    plt.show()
    '''
    cv2.imwrite(out, image_result)

    return image_result

images = glob.glob("samples/*.jpg")
i = 0
for image in images:
    process_image(image, "a{0}.jpg".format(i))
    i += 1
