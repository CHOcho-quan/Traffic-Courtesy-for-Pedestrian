"""
This is the pseudocode of the framework
It will be rewrited by python follow

These cases are good and need not to wait for the pedestrians according to the rule
#1.If (Pedestrians are detected but not overstepping the lane line)
#2.If (Pedestrians are waiting out of the lane line)
#3.If (Pedestrians are moving cross the other side of the lane line)

These cases are bad and need to stop and wait for the pedestrians according to the rule
#1.If (Pedestrians overstepped the lane line of the first side)
#2.If (Pedestrians are waiting on the lane line that has Double Yellow lines)
"""
import matplotlib.pyplot as plt
import cv2
import PROTOTYPE.laneline_detection.utils
import imageio


# people contains the rectangle's x and y positions, simple example [(1, 30),(30, 1)]
def is_courtesy(image, lines, people):
    imshape = image.shape
    ml, bl, mr, br = PROTOTYPE.laneline_detection.utils.get_lines(lines, imshape)
    peoplex = (people[0][0] + people[1][0]) / 2
    peopley1 = people[0][1]
    peopley2 = people[1][1]
    cv2.line(image, (peoplex, peopley1), (peoplex, peopley2), (0, 255, 0), 5)

    '''
    lane line equation: y = mx + b
    if the line intersect with the line represents people
    then we need to wait according to the rule
    '''
    # considering if we need to wait. To add situations by if, please return True
    if mr * peoplex + br < max(peopley1, peopley2):
        print 1
        if ml * peoplex + bl < max(peopley1, peopley2):
            print 2
            cv2.putText(image, "Please Wait!", (0, imshape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            return True

    # considering if we need not to wait
    print 3
    cv2.putText(image, "It's OK!", (0, imshape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    return False
