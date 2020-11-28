import cv2 as cv
import pickle
import os
import glob
import copy
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from prep.menu import FunctionItem
from prep.film import Film


class Database:

    DB_DIR = "prep/db/"
    SAVE_EXTENSION = ".png"
    PERSISTENCE_FILE = "db.p"

    def __init__(self):
        self.data = []
        self._load()

    def add_image(self):
        root = Tk()
        root.withdraw()
        filename = askopenfilename(
            filetypes=[("Image Files", ".png .jpg .jpeg")])
        root.destroy()

        img_name = filename.split("/")[-1].split(".")[-2]

        img = cv.imread(filename)

        film_name = input(" Film Name >> ")
        film_score = int(input(" Film Score >> "))

        cv.imwrite(self.DB_DIR + film_name + self.SAVE_EXTENSION, img)

        keypoints, descriptors = self._extract_features(img)

        film = Film(film_name, film_score, keypoints, descriptors)
        self.data.append(film)

        self._save()

    def list_images(self):
        items = []
        for film in self.data:
            item = FunctionItem(film.title, self.remove_image, [film.title])
            items.append(item)

        return items

    def remove_image(self, film_name):
        image = glob.glob(f"{self.DB_DIR}{film_name}.*")[0]

        try:
            os.remove(image)
        except:
            pass

        self._remove_film(film_name)

    def get_films_with_images(self):
        films_with_images = []

        for film in self.data:
            img_path = glob.glob(f"{self.DB_DIR}{film.title}.*")[0]
            img = cv.imread(img_path)

            films_with_images.append((film, img))

        return films_with_images

    def _remove_film(self, title):
        for film in self.data:
            if film.title == title:
                self.data.remove(film)

        self._save()

    def _load(self):
        try:
            data = pickle.load(open(self.DB_DIR + self.PERSISTENCE_FILE, "rb"))

            for film in data:
                film.keypoints = self._deserialize_keypoints(film.keypoints)

            self.data = data
        except FileNotFoundError:
            pass

    def _save(self):
        data = self._copy_data()

        for film in data:
            film.keypoints = self._serialize_keypoints(film.keypoints)

        pickle.dump(data, open(self.DB_DIR + self.PERSISTENCE_FILE, "wb"))

    def _extract_features(self, img):
        minHessian = 400
        detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        keypoints, descriptors = detector.detectAndCompute(
            cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)

        return keypoints, descriptors

    def _serialize_keypoints(self, keypoints):
        ser_keypoints = []

        for keypoint in keypoints:
            ser_keypoints.append((keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave,
                                  keypoint.class_id))

        return ser_keypoints

    def _deserialize_keypoints(self, ser_keypoints):
        keypoints = []

        for keypoint in ser_keypoints:
            keypoints.append(
                cv.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], _size=keypoint[1], _angle=keypoint[2],
                            _response=keypoint[3], _octave=keypoint[4], _class_id=keypoint[5]))

        return keypoints

    def _copy_data(self):
        data = []

        for film in self.data:
            copy = Film(film.title, film.score, film.keypoints, film.descriptors)
            data.append(copy)

        return data
