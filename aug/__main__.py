import cv2 as cv
from aug.augmentation import Augmentation
from aug.options import Options
from aug.algorithms import *
from prep.menu import *

aug = Augmentation()
options = Options()

options_menu = Menu("Options")
change_mode_menu = Menu("Change Mode")
change_detector_menu = Menu("Change Algorithm")

normal_mode_menu_item = FunctionItem(
    "Set Normal Mode", options.change_mode, [False])
tutorial_mode_menu_item = FunctionItem(
    "Set Tutorial Mode", options.change_mode, [True])
change_mode_menu.add_item(normal_mode_menu_item)
change_mode_menu.add_item(tutorial_mode_menu_item)

surf_menu_item = FunctionItem("SURF", options.change_algorithm, [SURF()])
orb_menu_item = FunctionItem("ORB", options.change_algorithm, [ORB()])
brief_menu_item = FunctionItem("BRIEF", options.change_algorithm, [BRIEF()])
change_detector_menu.add_item(surf_menu_item)
change_detector_menu.add_item(orb_menu_item)
change_detector_menu.add_item(brief_menu_item)

disable_ransac_menu_item = FunctionItem("Use solvePnp", options.enable_solvePnpRansac, [False])
enable_ransac_menu_item = FunctionItem("Use solvePnpRansac", options.enable_solvePnpRansac, [True])

options_menu.add_menu(change_mode_menu)
options_menu.add_menu(change_detector_menu)
options_menu.add_item(disable_ransac_menu_item)
options_menu.add_item(enable_ransac_menu_item)

main_menu = Menu("Main Menu")
aug_menu_item = FunctionItem("Poster Augmentation", aug.start)

main_menu.add_item(aug_menu_item)
main_menu.add_menu(options_menu)
main_menu.add_item(ExitItem())

main_menu.show()
