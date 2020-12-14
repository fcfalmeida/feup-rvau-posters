from prep.database import *
from aug.algorithms import *

class Options(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)

            cls._instance.tutorial_mode = False
            cls._instance.algorithm = ORB()

        return cls._instance

    def change_mode(self, tutorial):
        self.tutorial_mode = tutorial

    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        Database().recompute_features()
