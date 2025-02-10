def main():
    verbose_dict = get_verbose_dict(False)
    gui = SplendorGUI(None)  # Create GUI first
    game = Game(verbose_dict, gui)  # Pass GUI to game
    gui.game = game  # Set game reference in GUI
    gui.run() 