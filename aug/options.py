class Options(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)

            cls._instance.tutorial_mode = False

        return cls._instance

    def change_mode(self, tutorial):
        self.tutorial_mode = tutorial
