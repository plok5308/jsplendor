from jsplendor.utils.config import Element, get_coin_comb

def get_action_description(action_idx, game):
    """Get human-readable description of an action"""
    if action_idx < 10:  # Take three different coins
        coin_ids = get_coin_comb(action_idx)
        colors = [Element(idx).name for idx in coin_ids]
        return f"Take three different coins: {', '.join(colors)}"
        
    elif action_idx < 15:  # Take two same coins
        color = Element(action_idx - 10).name
        return f"Take two {color} coins"
        
    elif action_idx < 27:  # Buy development card
        card_idx = action_idx - 15
        level = (card_idx // 4) + 1
        pos = card_idx % 4
        cards = (game.board.table_level1 if level == 1 else 
                game.board.table_level2 if level == 2 else 
                game.board.table_level3)
        if pos < len(cards) and cards[pos]:
            card = cards[pos]
            return f"Buy L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
        return f"Buy L{level} card at position {pos} (Invalid - no card)"
        
    elif action_idx < 30:  # Buy reserved card
        card_idx = action_idx - 27
        player = game.players[0]
        if card_idx < len(player.reserved_cards):
            card = player.reserved_cards[card_idx]
            return f"Buy reserved card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
        return f"Buy reserved card at position {card_idx} (Invalid - no card)"
        
    elif action_idx < 42:  # Reserve card
        card_idx = action_idx - 30
        level = (card_idx // 4) + 1
        pos = card_idx % 4
        cards = (game.board.table_level1 if level == 1 else 
                game.board.table_level2 if level == 2 else 
                game.board.table_level3)
        if pos < len(cards) and cards[pos]:
            card = cards[pos]
            return f"Reserve L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
        return f"Reserve L{level} card at position {pos} (Invalid - no card)"
        
    return "Unknown action"