import os
import logging
import sys
from datetime import datetime
import numpy as np
from jsplendor.utils.config import Element, get_coin_comb
from .base_logger import BaseLogger

class StreamToLogger:
    """
    Custom stream object that redirects stdout to both console and log file
    """
    def __init__(self, logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout

    def write(self, buf):
        # Write to original stdout (console)
        self.original_stdout.write(buf)
        # Write to log file
        for line in buf.rstrip().splitlines():
            if line.strip():  # Only log non-empty lines
                self.logger.info(line.rstrip())

    def flush(self):
        self.original_stdout.flush()

class TestLogger(BaseLogger):
    def __init__(self, log_dir, verbose=False):
        self.verbose = verbose
        self._setup_logger(log_dir)
        self.log_dir = log_dir

    def _setup_logger(self, log_dir):
        # Create logs directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'test_{timestamp}.log')
        self.log_file = log_file

        # Configure logging
        self.logger = logging.getLogger('jsplendor')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Always add console handler for statistics
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Prevent log propagation
        self.logger.propagate = False

    def info(self, message):
        """Log an info message"""
        if self.logger:
            self.logger.info(message)

    def log_config(self, args, model_path):
        """Log test configuration"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting test with configuration:")
        self.logger.info("-"*30)
        self.logger.info(f"Number of games: {args.num_games}")
        self.logger.info(f"Max steps per game: {args.max_steps}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info("="*50 + "\n")

    def log_game_result(self, game_n, steps, max_step):
        """Log individual game results"""
        if steps == max_step - 1:
            self.logger.info(f"Game {game_n}: Failed (exceeded max steps)")
        else:
            self.logger.info(f"Game {game_n}: Completed in {steps} steps")

    def log_statistics(self, results, game_n, max_step):
        """Log final statistics"""
        mean_steps = sum(results) / len(results)
        fail_count = sum(1 for result in results if result == max_step - 1)
        success_rate = (game_n - fail_count) / game_n * 100

        # Force print these statistics regardless of verbose setting
        self.logger.info("\n" + "="*50)
        self.logger.info("Final Test Results:")
        self.logger.info("-"*30)
        self.logger.info(f"Mean steps to completion: {mean_steps:.2f}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Failed games: {fail_count}/{game_n}")

        # Log detailed step distribution
        self.logger.info("\nDetailed Step Distribution:")
        
        # Count games under 20 steps
        count_under_20 = sum(1 for x in results if x < 20)
        self.logger.info(f"Games completed under 20 steps: {count_under_20}")
        
        # Count games for each step between 20-40
        for step in range(20, 41):
            count = sum(1 for x in results if x == step)
            self.logger.info(f"Games completed in exactly {step} steps: {count}")
        
        # Count games over 40 steps
        count_over_40 = sum(1 for x in results if x > 40)
        self.logger.info(f"Games completed in over 40 steps: {count_over_40}")

        self.logger.info(f"\nFull log saved to: {os.path.abspath(self.log_file)}")
        self.logger.info("="*50 + "\n")

        return mean_steps, success_rate, fail_count

    def __del__(self):
        # Restore original stdout when logger is destroyed
        # sys.stdout = self.original_stdout
        pass

class ActionLogger(BaseLogger):
    @staticmethod
    def log_action_probabilities(logger, env, action_probs, verbose=True):
        """Log available actions and their probabilities"""
        if not verbose:
            return
            
        valid_actions = np.where(env.get_action_mask())[0]
        logger.info("-" * 30)
        logger.info("Available actions:")
        
        # Create list of (action, prob, desc) tuples for sorting
        action_info = []
        
        # Get sum of probabilities for valid actions only
        valid_probs = action_probs[valid_actions]
        normalization_factor = valid_probs.sum()
        
        # Normalize probabilities to sum to 100%
        normalized_probs = valid_probs / normalization_factor * 100
        
        for i, valid_action in enumerate(valid_actions):
            prob = normalized_probs[i]
            
            if valid_action < 10:  # Original coin collection actions (0-9)
                coin_ids = get_coin_comb(valid_action)
                coins = [Element(ids).name for ids in coin_ids]
                desc = f"Get coins: {', '.join(coins)}"
            elif valid_action < 15:  # Double coin actions (10-14)
                color = Element(valid_action - 10).name
                desc = f"Get two {color} coins"
            else:  # Buy card actions (15-26)
                card_pos = valid_action - 15
                card = env.game.board.flatten_table_cards[card_pos]
                if card:
                    desc = f"Buy {card.name} (Level: {card.level}, VP: {card.victory_point}, Color: {card.gem_color})"
                else:
                    desc = "Buy card (empty slot)"
            action_info.append((valid_action, prob, desc))
        
        # Sort by action number in ascending order
        action_info.sort(key=lambda x: x[0])
        
        # Log all actions
        for action, prob, desc in action_info:
            logger.info(f"  Action {action}: {desc} ({prob:.1f}%)")
        
        # Log probability summary
        logger.info("-" * 30)
        logger.info(f"Total probability: 100.0%")

    @staticmethod
    def log_selected_action(logger, action, verbose=True):
        """Log the selected action"""
        if verbose:
            logger.info(f"Selected: Action {action}") 
