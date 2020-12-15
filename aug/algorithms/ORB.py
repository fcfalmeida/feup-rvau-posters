import cv2 as cv
from aug.algorithms.algorithm import Algorithm


class ORB(Algorithm):

    def __init__(self):
        super().__init__(cv.ORB_create(), cv.DescriptorMatcher_create(
            cv.DescriptorMatcher_BRUTEFORCE_HAMMING), 30)

    def __str__(self):
        return f"Detector: ORB\nMatcher: Bruteforce Hamming"
