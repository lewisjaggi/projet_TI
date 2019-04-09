import cv2 as cv
import numpy as np

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


if __name__ == '__main__':
    frame = get_first_frame('test/film/test2_1.mp4')
    ret, corners, mtx, dist = find_chess(frame)
    dist_pixel = get_dist_pixel(corners)
    cv.drawChessboardCorners(frame, (chessH, chessW), corners, 1)
    h, w = frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv.undistort(frame, mtx, dist, None)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imshow('camera', dst)
    cv.waitKey(0)
