class Film:

    def __init__(self, title, score, keypoints, descriptors):
        self.title = title
        self.score = score
        self.keypoints = keypoints
        self.descriptors = descriptors

    def __eq__(self, other):
        return self.title == other.title