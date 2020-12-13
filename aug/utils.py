from aug.options import Options


def tutorial_print(string):
    options = Options()
    if options.tutorial_mode:
        print(string)


def is_tutorial():
    options = Options()
    return options.tutorial_mode
