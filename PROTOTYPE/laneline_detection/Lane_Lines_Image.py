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

def process_image():
    # Read in and grayscale the image
    image = mpimg.imread('./test_pages/test2.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, plt.gray())
    plt.show()

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
    vertices = np.array([[(0, imshape[0]), (imshape[1] / 3, 1.7 * imshape[0] / 3),
                          (2 * imshape[1] / 3, 1.7 * imshape[0] / 3), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(canny, mask)
    plt.imshow(mask)
    plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # print(lines)

    # Iterate over the output "lines" and draw lines on a blank image
    left = (0, 0)
    right = (0, 0)
    leftnum = 0
    rightnum = 0

    norms = utils.hough_filter(lines)

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         theta = abs(math.atan((y1 - y2) / float(x1 - x2)) * 180 / math.pi)
    #         if (80 > theta) & (theta > 20):
    #             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    for t in norms.keys():
        for norm in norms[t]:
            for x1, x2, y1, y2 in norm[1:]:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

            '''
            fit_left = np.polyfit((x1, x2), (y1, y2), 1)

            if fit_left[0]>0:
                left=left+fit_left
                leftnum=leftnum+1

            else:
                right=right+fit_left
                rightnum=rightnum+1

    left=left/leftnum
    right=right/rightnum
    print(left,right)

    x1_b=int((imshape[0]-left[1])/left[0])
    x1_t=int((imshape[0]/2-left[1])/left[0])

    cv2.line(line_image,(x1_b,imshape[0]),(x1_t,imshape[0]/2),(255,0,0),5)

    x1_b=int((imshape[0]-right[1])/right[0])
    x1_t=int((imshape[0]/2-right[1])/right[0])

    cv2.line(line_image,(x1_b,imshape[0]),(x1_t,imshape[0]/2),(255,0,0),5)
    #plt.imshow(line_image)
    #plt.show()'''

    plt.imshow(line_image)
    plt.show()
    # Create a "color" binary image to combine with line image

    # color_edges = np.dstack((canny, canny, canny))

    # Draw the lines on the edge image

    image_result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(image_result)
    plt.show()

    return image_result


process_image()
