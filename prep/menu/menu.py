from prep.menu.menu_item import MenuItem
from prep.menu.back_item import BackItem

class Menu(MenuItem):

    def __init__(self, title, refresh_fun = None):
        self.title = title
        self.items = []
        self.refresh_fun = refresh_fun
        self.parent = None

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
            if self.refresh_fun is not None:
                self.items = self.refresh_fun()
                self.items.append(BackItem(self.parent)) 
        
            for i in range(len(self.items)):
                item = self.items[i]
                print("%2d - %s" % (i + 1, item.title))

            option = int(input("Select an Option > ")) - 1
            sel_item = self.items[option]
            sel_item.action()

    def action(self):
        self.show()

    def __eq__(self, value):
        return self.title == value.title
