from prep.menu.menu_item import MenuItem

class BackItem(MenuItem):

    def __init__(self, prev_menu):
        super().__init__("Back")
        self.prev_menu = prev_menu
    
    def action(self):
        self.prev_menu.show()