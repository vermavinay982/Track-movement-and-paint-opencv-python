# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:00:53 2019

@author: Vinay
"""

# import the necessary packages
from imutils.video import VideoStream

import datetime
import imutils
import time
import cv2
import numpy as np
 
# construct the argument parser and parse the arguments
min_area=10
firstFrame=None
vs = cv2.VideoCapture(0)
activity=0
#backSub = cv2.createBackgroundSubtractorMOG2([history, nmixtures, backgroundRatio[, noiseSigma]])
backSub = cv2.createBackgroundSubtractorMOG2(history=30)
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    ret,frame = vs.read()
    
    text = "Unoccupied"
    activity+=1
 
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame


    frameDelta = backSub.apply(gray)
    #frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        activity=0
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"


        

# draw the text and timestamp on the frame
    #cv2.putText(frame, "Room Status: "+text+" - "+str(activity), (10, 20),
        #cv2.FONT_HERSHEY_SIMPLEX( 0.5, (0, 0, 255), 2)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Feed", frame)
    #cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    
    
    # if(activity > 50):
        # black=np.zeros((len(frame),len(frame[0])))
        # cv2.imshow("Feed",black)
        
    
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()