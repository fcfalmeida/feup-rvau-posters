from aug.augmentation import Augmentation
from aug.options import Options
from prep.menu import *

aug = Augmentation()

options_menu = Menu("Options")
change_mode_menu = Menu("Change Mode")

normal_mode_menu_item = FunctionItem("Set Normal Mode", Options.changeMode, [False])
tutorial_mode_menu_item = FunctionItem("Set Tutorial Mode", Options.changeMode, [True])
change_mode_menu.add_item(normal_mode_menu_item)
change_mode_menu.add_item(tutorial_mode_menu_item)

options_menu.add_menu(change_mode_menu)

main_menu = Menu("Main Menu")
aug_menu_item = FunctionItem("Poster Augmentation", aug.start)

main_menu.add_item(aug_menu_item)
main_menu.add_menu(options_menu)
main_menu.add_item(ExitItem())

main_menu.show()