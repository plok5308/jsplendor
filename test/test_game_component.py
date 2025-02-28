from jsplendor.game import Board, Player
from jsplendor.card import get_all_development_cards, get_three_noble_cards
from jsplendor.coin import get_full_coin_for_board, get_empty_coin
from jsplendor.utils import TestLogger
import logging

def test_game_component():
    # Setup logger with basic configuration
    logger = TestLogger.get_logger()
    logger.setLevel(logging.INFO)
    
    # Add console handler if not present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info("Starting game component tests...")
    
    # Initialize components
    logger.info("Initializing game components...")
    all_development_cards = get_all_development_cards()
    selected_noble_cards = get_three_noble_cards()
    board_coins = get_full_coin_for_board()
    
    board = Board(
                name="board",
                development_cards=all_development_cards,
                noble_cards=selected_noble_cards,
                coins=board_coins,
                verbose=True,
                logger=logger)
    
    player1 = Player(
                name="player1",
                development_cards=[],
                noble_cards=[],
                coins=get_empty_coin(),
                verbose=True,
                logger=logger)
    
    player2 = Player(
                name="player2",
                development_cards=[],
                noble_cards=[],
                coins=get_empty_coin(),
                verbose=True,
                logger=logger)
    
    # Test initial states
    logger.info("\nTesting initial states...")
    assert len(board.noble_cards) == 3, "Board should have 3 noble cards"
    assert len(player1.development_cards) == 0, "Player 1 should start with no development cards"
    assert len(player2.development_cards) == 0, "Player 2 should start with no development cards"
    logger.info("Initial state tests passed")
    
    # Print status for verification
    logger.info("\nPrinting component states:")
    board.print_status(object_name="board")
    player1.print_status(object_name="player1")
    player2.print_status(object_name="player2")
    
    logger.info("\nAll game component tests completed successfully")

if __name__ == "__main__":
    test_game_component()


