import cv2 as cv
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from prep.menu import FunctionItem

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

        up_img = cv.imread(filename)
        right_img = cv.rotate(up_img, cv.ROTATE_90_CLOCKWISE)
        down_img = cv.rotate(up_img, cv.ROTATE_180)
        left_img = cv.rotate(up_img, cv.ROTATE_90_COUNTERCLOCKWISE)

        film_name = input("Film Name > ")
        film_score = int(input("Film Score > "))

        cv.imwrite(self.DB_DIR + film_name + "_up" + self.SAVE_EXTENSION, up_img)
        cv.imwrite(self.DB_DIR + film_name + "_right" + self.SAVE_EXTENSION, right_img)
        cv.imwrite(self.DB_DIR + film_name + "_down" + self.SAVE_EXTENSION, down_img)
        cv.imwrite(self.DB_DIR + film_name + "_left" + self.SAVE_EXTENSION, left_img)

        self.data.append((filename, film_name, film_score))

        self._save()

    def list_images(self):
        items = []
        for file_name, _, _ in self.data:
            item = FunctionItem(file_name, self.remove_image, [file_name])
            items.append(item)

        return items

    def remove_image(self, img_name):
        print(f"Remove img {img_name}")

    def _load(self):
        try:
            data = pickle.load(open(self.DB_DIR + self.PERSISTENCE_FILE, "rb"))
            self.data = data
        except FileNotFoundError:
            pass

    def _save(self):
        pickle.dump(self.data, open(self.DB_DIR + self.PERSISTENCE_FILE, "wb"))