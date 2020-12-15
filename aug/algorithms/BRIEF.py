import cv2 as cv
from aug.algorithms.algorithm import Algorithm


class BRIEF(Algorithm):

    def __init__(self):
        super().__init__(cv.xfeatures2d_StarDetector.create(),
                         cv.DescriptorMatcher_create(
                             cv.DescriptorMatcher_BRUTEFORCE_HAMMING), 30)

    def detect_and_compute(self, img):
        matcher=cv.xfeatures2d_BriefDescriptorExtractor.create()

        keypoints=self.detector.detect(img, None)
        keypoints, descriptors=matcher.compute(
            img, keypoints)

        return keypoints, descriptors

    def __str__(self):
        return f"Detector: BRIEF\nMatcher: {self.matcher}"