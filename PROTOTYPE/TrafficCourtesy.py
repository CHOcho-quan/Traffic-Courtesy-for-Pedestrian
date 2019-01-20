import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.models import Model
import struct
import cv2
import tensorflow as tf

from laneline_detection.Lane_Lines_Continuous import get_line_image, process_image
from laneline_detection import utils
from pedestrian_detection.KerasYolo3.yolo3 import *

if __name__ == "__main__":
    video = cv2.VideoCapture('./laneline_detection/test7.mp4')
    success, frame = video.read()
    if success:
        line_image = np.copy(frame) * 0

    writer = cv2.VideoWriter('./lalala2.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (1280, 720))

    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader("./pedestrian_detection/KerasYolo3/yolov3.weights")
    weight_reader.load_weights(yolov3)
    px = None
    py = None

    # last time's m and b, also the experienced answer
    lml = lbl = lmr = lbr = 0
    # last time's error of m and b
    xml = xbl = xmr = xbr = 5
    # this time's m and b
    imshape, ml, bl, mr, br = get_line_image(frame)
    errorness1 = 105
    errorness2 = 5
    line_image = utils.draw_lines_mb(ml, bl, mr, br, line_image, imshape)
    while success:
        """
        We get two lines left: ml, bl & right: mr, br
        Now we consider the relationship between them and denote whether
        the car should avoid for the traffic courtesy or not
        """
        if px is not None or py is not None:
            if max(py[0], py[1]) < frame.shape[1] * 3 / 8:
                print(1)
                cv2.putText(frame, "It's OK!", (0, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            else:
                peoplex = (px[0] + px[1]) / 2
                if mr * peoplex + br < max(py[0], py[1]) + 10:
                    if ml * peoplex + bl < max(py[0], py[1]) + 10:
                        print(2)
                        cv2.putText(frame, "Please Wait!", (0, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                    else:
                        print(3)
                        cv2.putText(frame, "It's OK!", (0, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                else:
                    print(4)
                    cv2.putText(frame, "It's OK!", (0, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        else:
            print(5)
            cv2.putText(frame, "It's OK!", (0, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        frame = process_image(frame, line_image)
        writer.write(frame)

        success, frame = video.read()
        lml = ml
        lbl = bl
        lmr = mr
        lbr = br
        imshape, ml, bl, mr, br = get_line_image(frame)

        if ml == 0:
            ml = lml
            bl = lbl
        if mr == 0:
            mr = lmr
            br = lbr

        Kg2ml = (errorness2**2 + xml**2) / (errorness2**2 + xml**2 + errorness1**2)
        ml = lml + (ml - lml) * Kg2ml**0.5
        xml = 5 * (1 - Kg2ml)**0.5

        Kg2bl = (errorness2 ** 2 + xbl ** 2) / (errorness2 ** 2 + xbl ** 2 + errorness1 ** 2)
        bl = lbl + (bl - lbl) * Kg2bl ** 0.5
        xbl = 5 * (1 - Kg2bl) ** 0.5

        Kg2mr = (errorness2 ** 2 + xmr ** 2) / (errorness2 ** 2 + xmr ** 2 + errorness1 ** 2)
        mr = lmr + (mr - lmr) * Kg2mr ** 0.5
        xmr = 5 * (1 - Kg2mr) ** 0.5

        Kg2br = (errorness2 ** 2 + xbr ** 2) / (errorness2 ** 2 + xbr ** 2 + errorness1 ** 2)
        br = lbr + (br - lbr) * Kg2br ** 0.5
        xbr = 5 * (1 - Kg2br) ** 0.5

        image_h, image_w, _ = frame.shape
        new_image = preprocess_input(frame, net_h, net_w)
        yolos = yolov3.predict(new_image)
        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        # draw bounding boxes on the image using labels
        _, px, py = draw_boxes(frame, boxes, labels, obj_thresh)

        frame = (frame).astype('uint8')

        line_image = np.copy(frame) * 0
        line_image = utils.draw_lines_mb(ml, bl, mr, br, line_image, imshape)
