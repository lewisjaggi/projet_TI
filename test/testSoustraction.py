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
	

if __name__ == '__main__':
    #substract1()
	substract2()
	#substract3()
