from jsplendor.game import Board
from jsplendor.card import get_all_development_cards, get_three_noble_cards
from jsplendor.coin import get_full_coin_for_board
from jsplendor.utils import TestLogger
import logging

def test_board():
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
    
    logger.info("Starting board tests...")
    
    # Initialize components
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
    
    logger.info("Testing laying level 1 cards...")
    # Test laying level 1 cards
    res_sum = 0
    for i in range(40):
        res = board.lay_level_card_on_table(level=1)
        res_sum += res
    assert res_sum == -1 * board.level1_n, f"Expected {-1 * board.level1_n}, got {res_sum}"
    logger.info("Level 1 cards test passed")

    logger.info("Testing laying level 2 cards...")
    # Test laying level 2 cards
    res_sum = 0
    for i in range(30):
        res = board.lay_level_card_on_table(level=2)
        res_sum += res
    assert res_sum == -1 * board.level2_n, f"Expected {-1 * board.level2_n}, got {res_sum}"
    logger.info("Level 2 cards test passed")

    # Test flatten table cards
    logger.info("Testing flattened table cards...")
    assert len(board.flatten_table_cards) == 12, f"Expected 12 table cards, got {len(board.flatten_table_cards)}"
    logger.info("Flattened table cards test passed")

    # Print final board state
    logger.info("\nFinal board state:")
    board.print_status()

    logger.info("\nAll board tests completed successfully")

if __name__ == "__main__":
    test_board()