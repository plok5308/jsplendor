from jsplendor.game import Game
from jsplendor.gui.game_gui import SplendorGUI
from jsplendor.utils import get_verbose_dict

def main():
    verbose_dict = get_verbose_dict(False)
    game = Game(verbose_dict)
    gui = SplendorGUI(game)
    gui.run()

if __name__ == "__main__":
    main() 