import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class Database:

    DB_DIR = "prep/db/"
    SAVE_EXTENSION = ".png"
    
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

        cv.imwrite(self.DB_DIR + img_name + "_up" + self.SAVE_EXTENSION, up_img)
        cv.imwrite(self.DB_DIR + img_name + "_right" + self.SAVE_EXTENSION, right_img)
        cv.imwrite(self.DB_DIR + img_name + "_down" + self.SAVE_EXTENSION, down_img)
        cv.imwrite(self.DB_DIR + img_name + "_left" + self.SAVE_EXTENSION, left_img)