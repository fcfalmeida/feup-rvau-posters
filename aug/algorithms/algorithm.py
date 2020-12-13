class Algorithm:

    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

    def detect_and_compute(self, img):
        keypoints = self.detector.detect(img, None)
        keypoints, descriptors = self.detector.compute(
            img, keypoints)

        return keypoints, descriptors

    def get_matches(self, img1_descriptors, img2_descriptors):
        knn_matches = self.matcher.knnMatch(
            img1_descriptors, img2_descriptors, 2)

        return knn_matches
