import os
import logging
import sys
from datetime import datetime
import numpy as np
from jsplendor.utils.config import Element, get_coin_comb
from .base_logger import BaseLogger


class TestLogger(BaseLogger):
    _instance = None
    
    @classmethod
    def get_logger(cls, log_dir='logs/game'):
        if cls._instance is None:
            cls._instance = cls(log_dir)
        return cls._instance

    def __init__(self, log_dir, verbose=True):
        self.verbose = verbose
        self._setup_logger(log_dir)
        self.log_dir = log_dir
        print(f"Log file created at: {self.log_file}")

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

        # Console handler
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