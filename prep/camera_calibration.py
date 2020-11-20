import cv2 as cv
import numpy as np
import time


class CameraCalibration:
    # Number of images that should be captured for calibration
    NUM_CALIB_IMAGES = 10
    # Time interval between each image capture (seconds)
    CAPTURE_INTERVAL = 1

    def __init__(self, cbrows=9, cbcols=6):
        self.cbrows = cbrows
        self.cbcols = cbcols
        self.objpoints = []
        self.imgpoints = []
        self.criteria = (cv.TERM_CRITERIA_EPS +
                         cv.TERM_CRITERIA_MAX_ITER, self.NUM_CALIB_IMAGES, 0.001)
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def start(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.cbrows * self.cbcols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cbrows, 0:self.cbcols].T.reshape(-1, 2)

        img_count = 0  # number of valid calibration images collected

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        t = time.time()

        while True:
            # Capture frame-by-frame
            success, frame = cap.read()
            # if frame is read correctly ret is True
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if time.time() - t >= self.CAPTURE_INTERVAL:
                if img_count == self.NUM_CALIB_IMAGES:
                    break

                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(
                    gray_frame, (self.cbrows, self.cbcols), None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    img_count += 1
                    print("Captured image %d of %d" %
                          (img_count, self.NUM_CALIB_IMAGES))

                    self.objpoints.append(objp)
                    corners2 = cv.cornerSubPix(
                        gray_frame, corners, (11, 11), (-1, -1), self.criteria)
                    self.imgpoints.append(corners)

                    cv.drawChessboardCorners(
                        frame, (self.cbrows, self.cbcols), corners2, ret)

                t = time.time()

            cv.imshow('Calibration', frame)
            cv.waitKey(1)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, gray_frame.shape[::-1], None, None)
