import numpy as np
from gymnasium import spaces
from jsplendor.utils import Element

HIGH_VALUE = 63

class BoardObservation:
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
        coin_space = 6
        card_space = 4 * 3 * 7
        noble_space = 5 * 7
        total_space = coin_space + card_space + noble_space
        
        return total_space
    
    @staticmethod
    def get_observation(board):
        coin_obs = BoardObservation.get_coin_obs(board)
        card_obs = BoardObservation.get_card_obs(board)
        noble_obs = BoardObservation.get_noble_obs(board)
        
        obs = np.concatenate([coin_obs, card_obs, noble_obs])
        return np.clip(obs, 0, HIGH_VALUE-2)
    
    @staticmethod
    def get_coin_obs(board):
        x = np.zeros(6, dtype=np.int32)
        if isinstance(board.coins, dict):
            for key, value in board.coins.items():
                if isinstance(key, str):
                    x[BoardObservation.COLOR_MAP[key]] = value
                else:
                    x[key] = value
        else:
            for i, value in enumerate(board.coins):
                x[i] = value
        return x
    
    @staticmethod
    def get_card_obs(board):
        card_obs = []
        for card in board.flatten_table_cards:
            if card is not None:
                card_obs.extend(BoardObservation._get_card_features(card))
            else:
                card_obs.extend([0] * 7)
        return np.array(card_obs, dtype=np.int32)
    
    @staticmethod
    def get_noble_obs(board):
        noble_obs = []
        for noble in board.noble_cards:
            if noble is not None:
                noble_obs.extend(BoardObservation._get_noble_features(noble))
            else:
                noble_obs.extend([0] * 7)
        noble_obs.extend([0] * (35 - len(noble_obs)))
        return np.array(noble_obs, dtype=np.int32)
    
    @staticmethod
    def _get_card_features(card):
        return [
            card.victory_point,
            BoardObservation.COLOR_MAP[card.gem_color],
            card.price[0],
            card.price[1],
            card.price[2],
            card.price[3],
            card.price[4]
        ]
    
    @staticmethod
    def _get_noble_features(noble):
        return [
            noble.victory_point,
            0,
            noble.price[0],
            noble.price[1],
            noble.price[2],
            noble.price[3],
            noble.price[4]
        ]

class PlayerObservation:
    COLOR_MAP = BoardObservation.COLOR_MAP
    
    @staticmethod
    def get_space():
        step_space = 1
        coin_space = 6
        development_space = 5
        victory_point_space = 1
        level1_count_space = 1
        reserved_cards_space = 3 * 7
        price_diff_space = 12 * 5
        total_cost_space = 12 * 5
        
        total_space = (step_space + coin_space + development_space + 
                      victory_point_space + level1_count_space +
                      reserved_cards_space + price_diff_space + total_cost_space)
        
        return total_space
    
    @staticmethod
    def get_level1_count_obs(player):
        count = sum(1 for card in player.development_cards if card.level == 1)
        return np.array([min(count, HIGH_VALUE-2)], dtype=np.int32)
    
    @staticmethod
    def get_observation(game, player):
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
            level1_count_obs,
            price_diff_obs,
            total_cost_obs,
            reserved_cards_obs
        ])
        
        return np.clip(obs, 0, HIGH_VALUE-2)

    @staticmethod
    def get_step_obs(player):
        x = np.zeros(1, dtype=np.int32)
        x[0] = min(player.step, HIGH_VALUE-2)
        return x
    
    @staticmethod
    def get_coin_obs(player):
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
        x = np.zeros(5, dtype=np.int32)
        for card in player.development_cards:
            x[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        return np.clip(x, 0, HIGH_VALUE-2)
    
    @staticmethod
    def get_victory_point_obs(player):
        x = np.zeros(1, dtype=np.int32)
        x[0] = min(player.sum_victory_point, HIGH_VALUE-2)
        return x

    @staticmethod
    def get_card_price_diff_obs(board, player):
        cards = board.flatten_table_cards
        player_gems = np.zeros(5, dtype=np.int32)
        
        for card in player.development_cards:
            player_gems[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        
        diffs = []
        for card in cards:
            if card is None:
                diffs.extend([0] * 5)
            else:
                card_price = np.array(card.price)
                diff = card_price - player_gems
                diff = np.clip(diff, 0, HIGH_VALUE-2)
                diffs.extend(diff)
        
        return np.array(diffs, dtype=np.int32)

    @staticmethod
    def get_total_cost_obs(board, player):
        cards = board.flatten_table_cards
        player_gems = np.zeros(5, dtype=np.int32)
        
        player_coins = np.zeros(5, dtype=np.int32)
        for color, count in player.coins.items():
            if color != 'GOLD':
                player_coins[PlayerObservation.COLOR_MAP[color]] = count
        
        for card in player.development_cards:
            player_gems[PlayerObservation.COLOR_MAP[card.gem_color]] += 1
        
        total_costs = []
        for card in cards:
            if card is None:
                total_costs.extend([0] * 5)
            else:
                card_price = np.array(card.price)
                cost = card_price - player_gems - player_coins
                cost = np.clip(cost, 0, HIGH_VALUE-2)
                total_costs.extend(cost)
        
        return np.array(total_costs, dtype=np.int32)

    @staticmethod
    def get_reserved_cards_obs(player):
        x = np.zeros(3 * 7, dtype=np.int32)
        for i, card in enumerate(player.reserved_cards):
            x[i * 7:(i + 1) * 7] = BoardObservation._get_card_features(card)
        return x

def get_observation_space(game):
    board_space = BoardObservation.get_space()
    player_space = PlayerObservation.get_space()
    action_space = game.players[0].num_actions
    
    total_space = (board_space + (player_space * 2))*2 + action_space
    
    return spaces.Box(
        low=0,
        high=HIGH_VALUE,
        shape=(total_space,),
        dtype=np.int32
    )

def get_observation(previous_game, game, player_idx=0, verbose=False, logger=None):

    if previous_game is None:
        previous_obs = np.zeros(BoardObservation.get_space() + PlayerObservation.get_space()*2, dtype=np.int32)
    else:
        previous_board_obs = BoardObservation.get_observation(previous_game.board)
        previous_current_player = previous_game.players[player_idx]
        previous_opponent_player = previous_game.players[1 - player_idx]
        previous_current_obs = PlayerObservation.get_observation(previous_game, previous_current_player)
        previous_opponent_obs = PlayerObservation.get_observation(previous_game, previous_opponent_player)
        
        previous_obs = np.concatenate([
            previous_board_obs,
            previous_current_obs,
            previous_opponent_obs
        ])

    current_board_obs = BoardObservation.get_observation(game.board)
    current_player = game.players[player_idx]
    current_opponent_player = game.players[1 - player_idx]
    current_obs = PlayerObservation.get_observation(game, current_player)
    current_opponent_obs = PlayerObservation.get_observation(game, current_opponent_player)
    
    current_obs = np.concatenate([
        current_board_obs,
        current_obs,
        current_opponent_obs
    ])  

    obs = np.concatenate([
        previous_obs,
        current_obs,
    ])

    
    
    return obs

def get_high_level_price_sum_obs(board):
    total_price = np.zeros(5, dtype=np.int32)
    
    for card in board.flatten_table_cards:
        if card is not None and card.level in [2, 3]:
            total_price += np.array(card.price)
    
    total_price = np.clip(total_price, 0, HIGH_VALUE-2)
    
    return total_price
