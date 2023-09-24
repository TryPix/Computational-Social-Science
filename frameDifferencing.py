"""
Frame differencing to detect motion using opencv. 

Video source: https://www.pexels.com/video/aerial-view-of-bridge-and-river-2292093/

Saimaneesh Yeturu - 09/2023

"""

import cv2
import numpy as np

capture_cam = cv2.VideoCapture("pexels_videos_2292093 (2160p).mp4"); 

nFrames = capture_cam.get(cv2.CAP_PROP_FRAME_COUNT)

ret, curr_frame = capture_cam.read();
prev_frame = curr_frame;

for i in range(0, round(nFrames)):

    curr_frame_grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY); # make frames greyscale
    prev_frame_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY);


    # absolute difference for each pixel (this works as the frames are in greyscale)
    frame_diff = cv2.absdiff(curr_frame_grey, prev_frame_grey); 

    kernel = np.ones((5,5))

    # Erosion (erode bounderies of object), then dilation (increase holes, push whites to foreground): 
    frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel)

    # Lower threshold; convert each pixel to either black or white. This clearly seperates moving objects from non-moving
    thresh_frame = cv2.threshold(src=frame_diff, thresh=30, maxval=255, type=cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    # Draw contours on current (color) frame
    cv2.drawContours(image=curr_frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # Display frame in a window
    cv2.imshow('Frame Differencing', curr_frame)

    # Or display thresh_frame!
    # cv2.imshow('Frame Differencing', thresh_frame)

    key = cv2.waitKey(10); # wait
    if (key == ord('q')): # to exit
        break;

    prev_frame = curr_frame.copy(); # move to next frame
    ret, curr_frame = capture_cam.read();

capture_cam.release();
cv2.destroyAllWindows();


