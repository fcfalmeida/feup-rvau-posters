from prep.menu.menu_item import MenuItem

class FunctionItem(MenuItem):

    def __init__(self, title, function):
        super().__init__(title)
        self.function = function

    def action(self):
        self.function()