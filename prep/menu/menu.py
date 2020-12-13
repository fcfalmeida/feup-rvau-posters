import platform
import subprocess
from prep.menu.menu_item import MenuItem
from prep.menu.back_item import BackItem
from prep.menu.menu_borders import MenuBorders


class Menu(MenuItem):

    def __init__(self, title, refresh_fun=None):
        self.title = title
        self.items = []
        self.refresh_fun = refresh_fun
        self.parent = None
        self.borders = MenuBorders()

    def add_item(self, item):
        self.items.append(item)

    def add_items(self, items):
        self.items.extend(items)

    def add_menu(self, menu):
        menu.set_parent(self)
        self.items.append(menu)

    def set_parent(self, parent):
        self.parent = parent
        self.items.append(BackItem(parent))

    def show(self):
        while True:
            self.clear()

            if self.refresh_fun is not None:
                self.items = self.refresh_fun()
                self.items.append(BackItem(self.parent))

            # Menu Top
            print(self.borders.top_left_corner, end='')
            print(self.borders.horizontal_border * 39, end='')
            print(self.borders.top_right_corner)
            print("{0} {1} {2}".format(self.borders.vertical_border,
                                       self.title.center(37), self.borders.vertical_border))
            print("{0} {1} {2}".format(self.borders.vertical_border,
                                       '='*37, self.borders.vertical_border))
            print("{0} {1:37} {2}".format(self.borders.vertical_border,
                                          '', self.borders.vertical_border))

            # Menu Items
            for i in range(len(self.items)):
                item = self.items[i]
                print(
                    "{0} {1:2} - {2:33}{0}".format(self.borders.vertical_border, i+1, item.title))

            # Menu Bottom
            print("{0:39} {1}".format(
                self.borders.vertical_border, self.borders.vertical_border))
            print(self.borders.bottom_left_corner, end='')
            print(self.borders.horizontal_border * 39, end='')
            print(self.borders.bottom_right_corner)

            try:
                option = int(input(" Select an Option >> ")) - 1
                sel_item = self.items[option]
                sel_item.action()
            except (IndexError, ValueError):
                print("Invalid Option")

    def clear(self):
        if platform.system() == 'Windows':
            subprocess.check_call('cls', shell=True)
        else:
            print(subprocess.check_output('clear').decode())

    def action(self):
        self.show()

    def __eq__(self, value):
        return self.title == value.title
