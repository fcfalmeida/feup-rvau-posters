import cv2 as cv
import sys
import time
import numpy as np
from arposters.camera_calibration import CameraCalibration

calib = CameraCalibration()
calib.start()