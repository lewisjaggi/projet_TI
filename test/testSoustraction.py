# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import imutils
	
def substract1():
	src1 = cv.imread("images/image1.png")
	src2 = cv.imread("images/image4.png")
	result = src1-src2
	resultS = cv.resize(result, (960, 740))  
	cv.imshow("test", resultS)
	cv.waitKey(0)

def substract2():
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
	#fgbg = cv.createBackgroundSubtractorMOG2()

	src1 = cv.imread("images/image1.png")
	src2 = cv.imread("images/image4.png")
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 740))  
	cv.imshow("test", resultS)
	cv.waitKey(0)
	
def substract3():

	src1 = cv.imread("images/image1.png")
	src2 = cv.imread("images/image4.png")
	result = src1-src2
	cv.fastNlMeansDenoisingColored(result,result)
	
	resultS = cv.resize(result, (960, 740))
	cv.imshow("test", resultS)
	cv.waitKey(0)
	
def useCamera():
	vidcap = cv.VideoCapture('film/test4.mp4')
	success,image1 = vidcap.read()

	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	while True:   
		success,image2 = vidcap.read()

		if not success:
			break

		result = image1-image2
		fgmask = fgbg.apply(image1)
		fgmask = fgbg.apply(image2)
		
		resultS = cv.resize(fgmask, (960, 740))  
		cv.imshow("test", resultS)
		cv.waitKey(10)

		image1 = image2

def findgravityCenter(image1,image2):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	src1 = image1
	src2 = image2
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 660))  

	blurred = cv.GaussianBlur(resultS, (5, 5), 0)
	thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]
	M = cv.moments(thresh)
	
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
	cv.circle(thresh, (cX, cY), 5, (100, 100, 100), -1)

	return thresh

def findgravityCenterMultipleBlob(image1, image2):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	src1 = image1
	src2 = image2
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 740))

	blurred = cv.GaussianBlur(resultS, (5, 5), 0)
	thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]

	im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
	for c in contours:
		M = cv.moments(c)
		
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		cv.circle(thresh, (cX, cY), 5, (100, 100, 100), -1)

	return thresh

def findBasicCenterMultiple(image1, image2):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	src1 = image1
	src2 = image2
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 740))

	blurred = cv.GaussianBlur(resultS, (5, 5), 0)
	thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]

	cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		M = cv.moments(c)
		
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		cv.circle(thresh, (cX, cY), 5, (100, 100, 100), -1)
	
	return thresh

def findBasicCenterSingle(image1, image2):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	src1 = image1
	src2 = image2
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 740))

	blurred = cv.GaussianBlur(resultS, (5, 5), 0)
	thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]

	cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	M = cv.moments(cnts)
	
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
	cv.circle(thresh, (cX, cY), 5, (100, 100, 100), -1)
	
	return thresh

def findBasicCenterMultipleWithErosionDilation(image1, image2):
	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	src1 = image1
	src2 = image2
	
	fgmask = fgbg.apply(src1)
	fgmask = fgbg.apply(src2)
	resultS = cv.resize(fgmask, (960, 740))

	blurred = cv.GaussianBlur(resultS, (5, 5), 0)
	thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]

	kernel = np.ones((10,10),np.uint8)
	erosion = cv.erode(thresh,kernel,iterations = 1)
	dilation = cv.dilate(erosion,kernel,iterations = 1)

	cnts = cv.findContours(dilation.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		M = cv.moments(c)
		
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		cv.circle(dilation, (cX, cY), 5, (100, 100, 100), -1)
	
	return dilation


def testGravity():
	vidcap = cv.VideoCapture('film/test2_3.mp4')
	success,image1 = vidcap.read()

	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	while True:   
		success,image1 = vidcap.read()

		if not success:
			break

		result = findBasicCenterMultipleWithErosionDilation(image1,image2)
		
		resultS = cv.resize(result, (960, 740))  
		cv.imshow("test", resultS)
		cv.waitKey(5)

		image1 = image2

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

def final():
	vidcap = cv.VideoCapture('film/test2_3.mp4')
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
    #substract1() # est juste pour voir la négation
	#substract2() # donne un bon résultat
	#substract3() # ne fonctionne pas vraiment
	#useCamera() # permet de voir la manipulation des images d'une caméra via substract2
	#testGravity()
	final()

# doc
# https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

