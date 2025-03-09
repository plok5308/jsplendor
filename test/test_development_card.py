from jsplendor.card import get_all_development_cards, get_level_development_cards
from jsplendor.utils import TestLogger
import logging

def test_development_card():
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
    
    logger.info("Starting development card tests...\n")
    
    # Test level 1 cards
    logger.info("Testing level 1 cards...")
    l1_cards = get_level_development_cards(level=1)
    assert len(l1_cards) == 40, f"Expected 40 level 1 cards, got {len(l1_cards)}"
    
    l1_price_sum = sum(sum(card.price) for card in l1_cards)
    l1_victory_point_sum = sum(card.victory_point for card in l1_cards)
    
    logger.info(f"Level 1 cards summary:")
    logger.info(f"- Total cards: {len(l1_cards)}")
    logger.info(f"- Total price: {l1_price_sum}")
    logger.info(f"- Total VP: {l1_victory_point_sum}")
    assert l1_victory_point_sum == 5, f"Expected 5 victory points, got {l1_victory_point_sum}"
    assert l1_price_sum == 165, f"Expected total price 165, got {l1_price_sum}"
    logger.info("Level 1 cards test passed\n")

    # Test level 2 cards
    logger.info("Testing level 2 cards...")
    l2_cards = get_level_development_cards(level=2)
    assert len(l2_cards) == 30, f"Expected 30 level 2 cards, got {len(l2_cards)}"
    
    l2_price_sum = sum(sum(card.price) for card in l2_cards)
    l2_victory_point_sum = sum(card.victory_point for card in l2_cards)
    
    logger.info(f"Level 2 cards summary:")
    logger.info(f"- Total cards: {len(l2_cards)}")
    logger.info(f"- Total price: {l2_price_sum}")
    logger.info(f"- Total VP: {l2_victory_point_sum}")
    assert l2_victory_point_sum == 55, f"Expected 55 victory points, got {l2_victory_point_sum}"
    assert l2_price_sum == 205, f"Expected total price 205, got {l2_price_sum}"
    logger.info("Level 2 cards test passed\n")

    # Test level 3 cards
    logger.info("Testing level 3 cards...")
    l3_cards = get_level_development_cards(level=3)
    assert len(l3_cards) == 20, f"Expected 20 level 3 cards, got {len(l3_cards)}"
    
    l3_price_sum = sum(sum(card.price) for card in l3_cards)
    l3_victory_point_sum = sum(card.victory_point for card in l3_cards)
    
    logger.info(f"Level 3 cards summary:")
    logger.info(f"- Total cards: {len(l3_cards)}")
    logger.info(f"- Total price: {l3_price_sum}")
    logger.info(f"- Total VP: {l3_victory_point_sum}")
    assert l3_victory_point_sum == 80, f"Expected 80 victory points, got {l3_victory_point_sum}"
    assert l3_price_sum == 215, f"Expected total price 215, got {l3_price_sum}"
    logger.info("Level 3 cards test passed\n")

    logger.info("All development card tests completed successfully")

if __name__ == "__main__":
    test_development_card()