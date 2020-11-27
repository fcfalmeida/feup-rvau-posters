import cv2 as cv
import pickle
import os
import glob
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
        filename = askopenfilename(filetypes=[("Image Files", ".png .jpg .jpeg")])
        root.destroy()

        img_name = filename.split("/")[-1].split(".")[-2]

        img = cv.imread(filename)

        film_name = input(" Film Name >> ")
        film_score = int(input(" Film Score >> "))

        cv.imwrite(self.DB_DIR + film_name + self.SAVE_EXTENSION, img)

        film = Film(film_name, film_score, None, None)
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

    def _remove_film(self, title):
        for film in self.data:
            if film.title == title:
                self.data.remove(film)

        self._save()

    def _load(self):
        try:
            data = pickle.load(open(self.DB_DIR + self.PERSISTENCE_FILE, "rb"))
            self.data = data
        except FileNotFoundError:
            pass

    def _save(self):
        pickle.dump(self.data, open(self.DB_DIR + self.PERSISTENCE_FILE, "wb"))
