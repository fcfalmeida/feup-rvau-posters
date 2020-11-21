from prep.menu.menu_item import MenuItem

class ExitItem(MenuItem):

    def __init__(self):
        super().__init__("Exit")
    
    def action(self):
        exit(0)