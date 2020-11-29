import cv2 as cv
import numpy as np
import time
import pickle
from tkinter import Tk
from prep.calibration_params import CalibrationParams

class CameraCalibration:
    # Number of images that should be captured for calibration
    NUM_CALIB_IMAGES = 10
    # Time interval between each image capture (seconds)
    CAPTURE_INTERVAL = 1
    # File in which calibration params are persisted
    PERSISTENCE_FILE = "calib.p"

    def __init__(self, cbrows=9, cbcols=6):
        self.cbrows = cbrows
        self.cbcols = cbcols
        self.objpoints = []
        self.imgpoints = []
        self.criteria = (cv.TERM_CRITERIA_EPS +
                         cv.TERM_CRITERIA_MAX_ITER, self.NUM_CALIB_IMAGES, 0.001)
        
        self.calibration_params = None

        self.root = None

        self._load()

    def start(self):
        self.root = Tk()
        self.root.withdraw()

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
                # Draw an indicator each time we try to capture an image of the chessboard
                cv.circle(frame, (20,20), 10, (0,0,255), -1)
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

            if cv.waitKey(1) == ord('q'):
                self._stop(cap)
                return

        # When everything done, release the capture
        self._stop(cap)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, gray_frame.shape[::-1], None, None)

        self.calibration_params = CalibrationParams(ret, mtx, dist)
        self._save()

        print(f"Camera calibration done. Parameters saved in {self.PERSISTENCE_FILE}")

    def _stop(self, cap):
        cap.release()
        cv.destroyAllWindows()
        # Prevents freezing when closing the window for some reason
        cv.waitKey(1)

    def _save(self):
        pickle.dump(self.calibration_params, open(self.PERSISTENCE_FILE, "wb"))

    def _load(self):
        try:
            self.calibration_params = pickle.load(open(self.PERSISTENCE_FILE, "rb"))
            print(self.calibration_params)
        except FileNotFoundError:
            pass

