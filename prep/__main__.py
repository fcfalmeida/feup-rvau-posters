from prep.camera_calibration import CameraCalibration
from prep.database import Database
from prep.menu import *

calib = CameraCalibration()
db = Database()

database_menu = Menu("Database")
add_image_menu_item = FunctionItem("Add Film Poster", db.add_image)
database_menu.add_item(add_image_menu_item)

main_menu = Menu("Main Menu")
calib_menu_item = FunctionItem("Calibrate Camera", calib.start)

main_menu.add_item(calib_menu_item)
main_menu.add_menu(database_menu)
main_menu.add_item(ExitItem())

main_menu.show()