# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import imutils

def findBallCenterFromColor(image, lower, upper):
	frame = cv.resize(image, (960, 740))
	blurred = cv.GaussianBlur(frame, (11, 11), 0)
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
 
	mask = cv.inRange(hsv, lower, upper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)

	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
 
	if len(cnts) > 0:
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		if radius > 10:
			cv.circle(frame, center, 5, (0, 0, 255), -1)
 
	return frame

def readVideo(video):
	vidcap = cv.VideoCapture(video)
	# HSV colors
	redLower = (0, 80, 30)
	redUpper = (60, 255, 255)

	while True:   
		success,image = vidcap.read()

		if not success:
			break

		result = findBallCenterFromColor(image,redLower,redUpper)
		
		resultS = cv.resize(result, (960, 740))  
		cv.imshow("test", resultS)
		cv.waitKey(5)

if __name__ == '__main__':
	readVideo('film/test2_3.mp4')

