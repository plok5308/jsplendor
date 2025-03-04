import random
import logging
import os
from datetime import datetime
from jsplendor.game import Game
from jsplendor.utils import TestLogger
from jsplendor.utils.config import get_verbose_dict

def setup_logging(log_dir='logs'):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'game_{timestamp}.log')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

def play_two_player_game(max_steps=100, verbose=True):
    # Setup logging
    logger, log_file = setup_logging()
    logger.info(f"Logging to: {log_file}\n")
    
    # Initialize game with verbose settings
    verbose_dict = get_verbose_dict(verbose)
    game = Game(verbose_dict=verbose_dict)
    
    # Add second player to game
    game.add_player("player2")
    
    step = 0
    current_player_idx = 0
    
    logger.info("\n=== Starting two player game ===\n")
    logger.info("Initial board state:")
    game.print_status()
    
    while step < max_steps:
        # Get current player
        current_player = game.players[current_player_idx]
        
        # Get valid actions for current player
        actions_bool = current_player.get_all_possible_actions(game.board)
        possible_actions = [i for i, valid in enumerate(actions_bool) if valid]
        
        if not possible_actions:
            logger.info(f"No valid actions for {current_player.name}")
            break
            
        # Choose random action
        action = random.choice(possible_actions)
        
        # Execute action
        victory_point, over_coin_count, got_card, noble_visit = current_player.do_action(game.board, action)
        
        # Print step information
        logger.info(f"\n=== Step {step + 1} ===")
        logger.info(f"{current_player.name}'s turn")
        logger.info(f"Action taken: {action}")
        logger.info(f"Victory points: {victory_point}")
        if got_card:
            logger.info("Got a development card")
        if noble_visit:
            logger.info("Got a noble card")
        if over_coin_count > 0:
            logger.info(f"Dropped {over_coin_count} coins")
            
        # Print updated game state
        logger.info("\nCurrent game state:")
        game.print_status()
        
        # Check victory condition (15 points)
        if victory_point >= 15:
            logger.info(f"\n{current_player.name} wins with {victory_point} points!")
            break
            
        # Switch players
        current_player_idx = (current_player_idx + 1) % 2
        step += 1
        
    if step >= max_steps:
        logger.info("\nGame ended due to maximum steps")
        # Determine winner
        points = [p.sum_victory_point for p in game.players]
        if points[0] > points[1]:
            logger.info(f"Player 1 wins with {points[0]} points! (Player 2: {points[1]} points)")
        elif points[1] > points[0]:
            logger.info(f"Player 2 wins with {points[1]} points! (Player 1: {points[0]} points)")
        else:
            logger.info(f"Game ended in a tie with {points[0]} points each!")
    
    logger.info(f"\nFull game log saved to: {log_file}")
    return game

if __name__ == "__main__":
    game = play_two_player_game(max_steps=100, verbose=True) 