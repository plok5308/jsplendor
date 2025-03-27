import numpy as np
from gymnasium import spaces
from jsplendor.utils import Element

HIGH_VALUE = 63

class BoardObservation:
    """Handles board state observations"""
    
    # Color mapping for consistent indexing
    COLOR_MAP = {
        'WHITE': 0,
        'BLUE': 1,
        'GREEN': 2,
        'RED': 3,
        'BLACK': 4,
        'GOLD': 5
    }
    
    @staticmethod
    def get_space():
        """Get observation space for board state"""
        coin_space = 6  # Available coins
        card_space = 4 * 3 * 7  # 4 cards x 3 levels x 7 features
        noble_space = 5 * 7  # 5 nobles x 7 features
        total_space = coin_space + card_space + noble_space  # 113
        
        return total_space
    
    @staticmethod
    def get_observation(board):
        """Get full board observation"""
        coin_obs = BoardObservation.get_coin_obs(board)
        card_obs = BoardObservation.get_card_obs(board)
        noble_obs = BoardObservation.get_noble_obs(board)
        
        obs = np.concatenate([coin_obs, card_obs, noble_obs])
        # Clip board observation values
        return np.clip(obs, 0, HIGH_VALUE-2)
    
    @staticmethod
    def get_coin_obs(board):
        """Get coin observation"""
        x = np.zeros(6, dtype=np.int32)
        if isinstance(board.coins, dict):
            # Handle dictionary format
            for key, value in board.coins.items():
                if isinstance(key, str):
                    x[BoardObservation.COLOR_MAP[key]] = value
                else:
                    x[key] = value
        else:
            # Handle list/array format
            for i, value in enumerate(board.coins):
                x[i] = value
        return x
    
    @staticmethod
    def get_card_obs(board):
        """Get card observation"""
        card_obs = []
        # Use flatten_table_cards to get all cards
        for card in board.flatten_table_cards:
            if card is not None:  # Handle empty slots
                card_obs.extend(BoardObservation._get_card_features(card))
            else:
                # Add zeros for empty slots
                card_obs.extend([0] * 7)  # 7 features per card
        return np.array(card_obs, dtype=np.int32)
    
    @staticmethod
    def get_noble_obs(board):
        """Get noble observation"""
        noble_obs = []
        # Use noble_cards directly
        for noble in board.noble_cards:
            if noble is not None:
                noble_obs.extend(BoardObservation._get_noble_features(noble))
            else:
                # Add zeros for empty slots
                noble_obs.extend([0] * 7)  # 7 features per noble
        # Pad to fixed size (5 nobles * 7 features = 35)
        noble_obs.extend([0] * (35 - len(noble_obs)))
        return np.array(noble_obs, dtype=np.int32)
    
    @staticmethod
    def _get_card_features(card):
        """Extract features from a development card"""
        return [
            card.victory_point,
            BoardObservation.COLOR_MAP[card.gem_color],  # Convert color string to index
            card.price[0],  # White
            card.price[1],  # Blue
            card.price[2],  # Green
            card.price[3],  # Red
            card.price[4]   # Black
        ]
    
    @staticmethod
    def _get_noble_features(noble):
        """Extract features from a noble card"""
        return [
            noble.victory_point,
            0,  # Noble's color (always white/0)
            noble.price[0],  # White - using price instead of requirement
            noble.price[1],  # Blue
            noble.price[2],  # Green
            noble.price[3],  # Red
            noble.price[4]   # Black
        ]

class PlayerObservation:
    """Handles player state observations"""
    
    # Share the same color mapping
    COLOR_MAP = BoardObservation.COLOR_MAP
    
    @staticmethod
    def get_space():
        """Get observation space for player state"""
        step_space = 1  # Current step
        coin_space = 6  # Player's coins
        development_space = 5  # Count of cards per color
        victory_point_space = 1  # Total victory points
        level1_count_space = 1  # Total count of level 1 cards
        reserved_cards_space = 3 * 7  # 3 reserved cards x card features
        
        # Add spaces for card price differences and total costs
        price_diff_space = 12 * 5  # 12 cards x 5 colors
        total_cost_space = 12 * 5  # 12 cards x 5 colors
        
        total_space = (step_space + coin_space + development_space + 
                      victory_point_space + level1_count_space +
                      reserved_cards_space + price_diff_space + total_cost_space)
        
        return total_space
    
    @staticmethod
    def get_level1_count_obs(player):
        """Get total count of level 1 cards"""
        count = sum(1 for card in player.development_cards if card.level == 1)
        return np.array([min(count, HIGH_VALUE-2)], dtype=np.int32)
    
    @staticmethod
    def get_observation(game, player):
        """Get full player observation including price differences and total costs"""
        step_obs = PlayerObservation.get_step_obs(player)
        coin_obs = PlayerObservation.get_coin_obs(player)
        development_obs = PlayerObservation.get_development_obs(player)
        victory_point_obs = PlayerObservation.get_victory_point_obs(player)
        level1_count_obs = PlayerObservation.get_level1_count_obs(player)
        price_diff_obs = PlayerObservation.get_card_price_diff_obs(game.board, player)
        total_cost_obs = PlayerObservation.get_total_cost_obs(game.board, player)
        reserved_cards_obs = PlayerObservation.get_reserved_cards_obs(player)
        
        obs = np.concatenate([
            step_obs, 
            coin_obs, 
            development_obs, 
            victory_point_obs,
            level1_count_obs,  # Add level 1 cards count
            price_diff_obs,
            total_cost_obs,
            reserved_cards_obs
        ])
        
        # Clip player observation values
        return np.clip(obs, 0, HIGH_VALUE-2)

    @staticmethod
    def get_step_obs(player):
        """Get step observation"""
        x = np.zeros(1, dtype=np.int32)
        x[0] = min(player.step, HIGH_VALUE-2)
        return x
    
    @staticmethod
    def get_coin_obs(player):
        """Get coin observation"""
        x = np.zeros(6, dtype=np.int32)
        if isinstance(player.coins, dict):
            for key, value in player.coins.items():
                if isinstance(key, str):
                    x[PlayerObservation.COLOR_MAP[key]] = min(value, HIGH_VALUE-2)
                else:
                    x[key] = min(value, HIGH_VALUE-2)
        else:
            for i, value in enumerate(player.coins):
                x[i] = min(value, HIGH_VALUE-2)
        return x
    
    @staticmethod
    def get_development_obs(player):
        """Get development card observation"""
        x = np.zeros(5, dtype=np.int32)
        for card in player.development_cards:
            x[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        return np.clip(x, 0, HIGH_VALUE-2)
    
    @staticmethod
    def get_victory_point_obs(player):
        """Get victory point observation"""
        x = np.zeros(1, dtype=np.int32)
        x[0] = min(player.sum_victory_point, HIGH_VALUE-2)
        return x

    @staticmethod
    def get_card_price_diff_obs(board, player):
        """Calculate difference between card prices and player's gems"""
        cards = board.flatten_table_cards
        player_gems = np.zeros(5, dtype=np.int32)
        
        # Count player's development cards by color
        for card in player.development_cards:
            player_gems[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        
        diffs = []
        for card in cards:
            if card is None:
                diffs.extend([0] * 5)
            else:
                # For each gem type, calculate price - player's gems
                card_price = np.array(card.price)
                diff = card_price - player_gems
                diff = np.clip(diff, 0, HIGH_VALUE-2)  # Clip values to valid range
                diffs.extend(diff)
        
        return np.array(diffs, dtype=np.int32)

    @staticmethod
    def get_total_cost_obs(board, player):
        """Calculate total cost considering both development cards and coins"""
        cards = board.flatten_table_cards
        player_gems = np.zeros(5, dtype=np.int32)
        
        # Convert player coins from dict to array
        player_coins = np.zeros(5, dtype=np.int32)
        for color, count in player.coins.items():
            if color != 'GOLD':  # Skip gold coins
                player_coins[PlayerObservation.COLOR_MAP[color]] = count
        
        # Count player's development cards by color
        for card in player.development_cards:
            player_gems[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        
        total_costs = []
        for card in cards:
            if card is None:
                total_costs.extend([0] * 5)
            else:
                # Calculate total cost: price - development cards - coins
                card_price = np.array(card.price)
                cost = card_price - player_gems - player_coins
                cost = np.clip(cost, 0, HIGH_VALUE-2)  # Clip values to valid range
                total_costs.extend(cost)
        
        return np.array(total_costs, dtype=np.int32)

    @staticmethod
    def get_reserved_cards_obs(player):
        """Get reserved card observation"""
        x = np.zeros(3 * 7, dtype=np.int32)
        for i, card in enumerate(player.reserved_cards):
            x[i * 7:(i + 1) * 7] = BoardObservation._get_card_features(card)
        return x

def get_observation_space(game):
    """Get total observation space including board, players, and action mask"""
    # Board space
    board_space = BoardObservation.get_space()  # 113 (6 coins + 84 cards + 35 nobles)
    
    # Player space (includes price diffs and total costs)
    player_space = PlayerObservation.get_space()  # Should match actual dimensions
    
    # CLS token space
    cls_space = 1  # 62 (using HIGH_VALUE=63)

    # Action space
    action_space = game.players[0].num_actions  # 43 actions including reserve
    
    # Total space calculation
    total_space = cls_space + board_space + (player_space * 2) + action_space
    
    # Print dimensions for debugging
    print(f"Observation space dimensions:")
    print(f" - CLS token: {cls_space}")
    print(f" - Board space: {board_space}")
    print(f" - Player space (x2): {player_space} * 2")
    print(f" - Action space: {action_space}")
    print(f" - Total space: {total_space}")
    
    return spaces.Box(
        low=0,
        high=HIGH_VALUE,
        shape=(total_space,),
        dtype=np.int32
    )

def get_observation(game, player_idx=0, verbose=False, logger=None):
    """Get full game observation from player's perspective"""
    # Get board observation
    board_obs = BoardObservation.get_observation(game.board)
    
    # Get player observations
    current_player = game.players[player_idx]
    opponent_player = game.players[1 - player_idx]
    current_obs = PlayerObservation.get_observation(game, current_player)
    opponent_obs = PlayerObservation.get_observation(game, opponent_player)
    
    # Add CLS token
    cls_token = np.ones(1, dtype=np.int32) * (HIGH_VALUE-1)
    
    # Combine observations
    obs = np.concatenate([
        cls_token,
        board_obs,
        current_obs,
        opponent_obs
    ])
    
    if verbose and logger:
        logger.info("-" * 50)
        logger.info("Base Observation:")
        logger.info(f"Shape: {obs.shape}")
        logger.info(f"CLS({cls_token.shape[0]}) + Board({board_obs.shape[0]}) + " +
                   f"Current({current_obs.shape[0]}) + Opponent({opponent_obs.shape[0]})")
    
    return obs

def get_high_level_price_sum_obs(board):
    """Sum of prices for level 2 and 3 cards per gem type"""
    total_price = np.zeros(5, dtype=np.int32)
    
    # Sum prices for level 2 and 3 cards
    for card in board.flatten_table_cards:
        if card is not None and card.level in [2, 3]:
            total_price += np.array(card.price)
    
    # Clip values to valid range
    total_price = np.clip(total_price, 0, HIGH_VALUE-2)
    
    return total_price
