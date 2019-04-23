# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import imutils
from math import sqrt

chessW = 8
chessH = 6
square_width = 4
square_height = 4


def get_first_frame(video_path):
    movie = cv.VideoCapture(video_path)
    if not movie.isOpened():
        print("Error opening video stream or file")

    ret, frame = movie.read()
    movie.release()
    cv.destroyAllWindows()
    return frame


def find_chess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    objpoints = []
    imgpoints = []  # 2d points in image plane.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessW * chessH, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessH, 0:chessW].T.reshape(-1, 2)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (chessH, chessW), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, corners, mtx, dist


def get_dist_pixel(corners):
    p1 = corners[0]
    p2 = corners[chessH - 1]
    p3 = corners[-1]
    p4 = corners[-chessH]
    height_real_pixel_1 = abs((p1 - p2)[0][1])
    width_real_pixel_1 = abs((p3 - p2)[0][0])

    diag1 = ((p1[0][0] - p3[0][0]) ** 2 + (p1[0][1] - p3[0][1]) ** 2) ** 0.5
    diag2 = ((p2[0][0] - p4[0][0]) ** 2 + (p2[0][1] - p4[0][1]) ** 2) ** 0.5
    diag_real = (((chessH - 1) * square_height) ** 2 + ((chessW - 1) * square_width) ** 2) ** 0.5

    height_real_pixel_2 = abs((p4 - p3)[0][1])
    width_real_pixel_2 = abs((p4 - p1)[0][0])

    l_pixel_by_cm = [diag1 / diag_real, diag2 / diag_real, width_real_pixel_1 / ((chessW - 1) * square_width),
                     width_real_pixel_2 / ((chessW - 1) * square_width),
                     height_real_pixel_1 / ((chessH - 1) * square_width),
                     height_real_pixel_2 / ((chessH - 1) * square_width)]

    return sum(l_pixel_by_cm) / len(l_pixel_by_cm)

def findBallCenterFromColor(image, lower, upper):
	blurred = cv.GaussianBlur(image, (11, 11), 0)
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
			return center
 
	return None

def readVideo(video, dist_pixel):
	vidcap = cv.VideoCapture(video)
	# HSV colors
	redLower = (0, 150, 70)
	redUpper = (25, 255, 255)

	previouscenter = None
	distance = 0
	vitesse = 0

	fps = vidcap.get(cv.CAP_PROP_FPS)
	
	while True:   
		success,image = vidcap.read()

		if not success:
			break

		center = findBallCenterFromColor(image,redLower,redUpper)

		if previouscenter and center:
			distance = sqrt((center[0] - previouscenter[0])**2 + (center[1] - previouscenter[1])**2) / dist_pixel
			vitesse = distance * fps
			cv.circle(image, center, 5, (0, 0, 255), -1)
			cv.putText(image,"{:.2f} cm/s".format(vitesse),(10,100), cv.FONT_HERSHEY_COMPLEX, 4, (255,255,255), 2, cv.LINE_AA)

		result = cv.resize(image, (960, 740))  
		cv.imshow("test", result)
		cv.waitKey(5)

		previouscenter = center

if __name__ == '__main__':
	video = 'test/film/test2_1.mp4'

	frame = get_first_frame(video)
	ret, corners, mtx, dist = find_chess(frame)
	dist_pixel = get_dist_pixel(corners)

	readVideo(video, dist_pixel)

