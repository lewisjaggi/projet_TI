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


def testGravity():
	vidcap = cv.VideoCapture('film/test2_2.mp4')
	success,image1 = vidcap.read()

	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

	while True:   
		success,image2 = vidcap.read()

		if not success:
			break

		result = findBasicCenterMultiple(image1,image2)
		
		resultS = cv.resize(result, (960, 740))  
		cv.imshow("test", resultS)
		cv.waitKey(10)

		image1 = image2

if __name__ == '__main__':
    #substract1() # est juste pour voir la négation
	#substract2() # donne un bon résultat
	#substract3() # ne fonctionne pas vraiment
	#useCamera() # permet de voir la manipulation des images d'une caméra via substract2
	testGravity()

# doc
# https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

