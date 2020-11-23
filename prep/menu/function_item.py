from prep.menu.menu_item import MenuItem

class FunctionItem(MenuItem):

    def __init__(self, title, function, args = []):
        super().__init__(title)
        self.function = function
        self.args = args # list

    def action(self):
        self.function(*self.args)