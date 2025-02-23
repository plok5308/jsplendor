import pygame
from jsplendor.game import Game
from jsplendor.gui.game_gui import AIGameGUI
from jsplendor.utils import get_verbose_dict, TestLogger
from jsplendor.env import JsplendorEnv
from stable_baselines3 import PPO

def main():
    # Enable verbose logging for all components
    verbose_dict = get_verbose_dict(False)
    verbose_dict['env'] = True
    verbose_dict['game'] = True
    verbose_dict['board'] = True
    verbose_dict['player'] = True
    
    # Initialize environment
    env = JsplendorEnv(verbose_dict)
    
    # Load model
    #exp = '250222_new_feat'
    #model_path = f'logs/{exp}/best_model'
    #log_dir = f'logs/{exp}/'
    model_path = 'pretrained/best_model'
    log_dir = 'pretrained/'
    
    # Initialize logger
    logger = TestLogger(log_dir)
    logger.info("Starting GUI test session")
    
    # Load model
    logger.info("Loading pretrained model...")
    model = PPO.load(model_path, env=env)
    logger.info('Model loaded successfully')
    
    # Create game instance
    game = Game(verbose_dict)
    
    # Create and run AI GUI
    app = AIGameGUI(game, env, model, logger)
    try:
        app.run()
    finally:
        logger.info("GUI test session ended")

if __name__ == "__main__":
    main() 
