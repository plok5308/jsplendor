import os
import logging
import sys
from datetime import datetime

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

class TestLogger:
    def __init__(self, exp_name):
        self.logger = self._setup_logger(exp_name)
        self.log_dir = os.path.join('logs', exp_name)
        
        # Save original stdout and redirect it
        self.original_stdout = sys.stdout
        self.stream_logger = StreamToLogger(self.logger, self.original_stdout)
        sys.stdout = self.stream_logger

    def __del__(self):
        # Restore original stdout when logger is destroyed
        sys.stdout = self.original_stdout

    def _setup_logger(self, exp_name):
        # Create logs directory if it doesn't exist
        self.log_dir = os.path.join('logs', exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'test_{timestamp}.log')

        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def log_config(self, args, model_path):
        """Log test configuration"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting test with configuration:")
        self.logger.info("-"*30)
        self.logger.info(f"Number of games: {args.num_games}")
        self.logger.info(f"Max steps per game: {args.max_steps}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info("="*50 + "\n")

    def log_game_result(self, game_id, steps, max_steps):
        """Log individual game results"""
        if steps < max_steps // 2:
            self.logger.info(f"Game {game_id} completed quickly in {steps} steps")
        elif steps == max_steps - 1:
            self.logger.info(f"Game {game_id} failed to complete within step limit")
        self.logger.info("-"*50)  # Add separator after each game

    def log_statistics(self, results, game_n, max_step):
        """Log final statistics"""
        mean_steps = sum(results) / len(results)
        fail_count = sum(1 for result in results if result == max_step - 1)
        success_rate = (game_n - fail_count) / game_n * 100

        self.logger.info("\n" + "="*50)
        self.logger.info("Final Test Results:")
        self.logger.info("-"*30)
        self.logger.info(f"Mean steps to completion: {mean_steps:.2f}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Failed games: {fail_count}/{game_n}")

        # Log detailed step distribution
        self.logger.info("\nStep Distribution:")
        step_bins = [0, 25, 50, 75, 100]
        for i in range(len(step_bins)-1):
            count = sum(1 for x in results if step_bins[i] <= x < step_bins[i+1])
            self.logger.info(f"Games completed in {step_bins[i]}-{step_bins[i+1]} steps: {count}")

        self.logger.info(f"\nFull log saved to: {self.log_dir}")
        self.logger.info("="*50 + "\n")

        return mean_steps, success_rate, fail_count

    def info(self, message):
        """Log an info message"""
        self.logger.info(message) 