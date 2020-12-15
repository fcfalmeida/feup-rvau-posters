import cv2 as cv
from aug.algorithms.algorithm import Algorithm


class SURF(Algorithm):

    def __init__(self):
        surf = cv.xfeatures2d_SURF.create(400)
        super().__init__(surf, cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED), 100)

    def __str__(self):
        return f"Detector: SURF\nMatcher: {self.matcher}"
