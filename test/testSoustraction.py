# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
	
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

if __name__ == '__main__':
    #substract1()
	#substract2()
	#substract3()
	useCamera()
