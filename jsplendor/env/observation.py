import numpy as np
from gymnasium import spaces

from jsplendor.game import Game
from jsplendor.utils import Element


HIGH_VALUE = 63

def get_observation_space():
    # Original space: 110
    # New features:
    # 1. Card price - player gems (12 cards * 5 gems = 60)
    # 2. Level2&3 total price per gem (5 gems)
    # Total: 110 + 60 + 5 = 175
    observation_space = spaces.Box(
        low=0,
        high=HIGH_VALUE,
        shape=(175,),  # Verify this matches actual observation size
        dtype=np.int32
    )
    return observation_space

def get_observation(game: Game):
    player1 = game.player1
    board = game.board

    obs_start = get_start_obs()
    obs0 = get_step_obs(game)
    obs1 = get_coin_obs(player1)
    obs2 = get_player_development_obs(player1)
    obs3 = get_player_victory_point_obs(player1)
    obs4 = get_table_coin_obs(board)
    obs5 = get_table_cards_obs(board)
    
    # New features
    obs6 = get_card_price_diff_obs(board, player1)
    obs7 = get_high_level_price_sum_obs(board)

    obs = np.concatenate([obs_start, obs0, obs1, obs2, obs3, obs4, obs5, obs6, obs7])
    obs = obs.astype(np.int32)
    
    # Add shape verification
    assert obs.shape[0] == 175, f"Expected observation size 175, got {obs.shape[0]}"
    
    return obs

def get_start_obs():
    x = np.zeros(1, dtype=np.int32)

    return x

def get_step_obs(game):
    x = np.zeros(1, dtype=np.int32)
    x[0] = min(game.step, HIGH_VALUE)

    return x

def get_coin_obs(player):
    coins = player.coins
    x = np.zeros(6, dtype=np.int32)
    for key, value in coins.items():
        x[Element[key].value] = value

    return x

def get_player_development_obs(player):
    cards = player.development_cards
    x = np.zeros(5, dtype=np.int32)
    for card in cards:
        x[Element[card.gem_color].value] += 1

    return x

def get_player_victory_point_obs(player):
    x = np.zeros(1, dtype=np.int32)
    x[0] = player.sum_victory_point

    return x

def get_table_coin_obs(board):
    coins = board.coins
    x = np.zeros(6, dtype=np.int32)
    for key, value in coins.items():
        x[Element[key].value] = value

    return x

def get_table_cards_obs(board):
    cards = board.flatten_table_cards

    for idx, card in enumerate(cards):
        if idx==0:
            obs = get_table_card_obs(card)
        else:
            obs_ = get_table_card_obs(card)
            obs = np.concatenate([obs, obs_])

    cards = board.noble_cards
    for card in cards:
        obs_ = get_table_card_obs(card)
        obs = np.concatenate([obs, obs_])

    return obs

def get_table_card_obs(card):
    if card is None:
        x = np.zeros(6, dtype=np.int32)
    else:
        x = np.zeros(6, dtype=np.int32)
        # Keep original integer values
        x[0:5] = np.array(card.price)  # 0-7
        x[5] = card.victory_point      # 0-20

    return x

def get_card_price_diff_obs(board, player):
    """Calculate difference between card prices and player's gems"""
    cards = board.flatten_table_cards
    player_gems = np.zeros(5, dtype=np.int32)
    
    # Count player's development cards by color
    for card in player.development_cards:
        player_gems[Element[card.gem_color].value] += 1
    
    diffs = []
    for card in cards:
        if card is None:
            diffs.extend([0] * 5)
        else:
            # For each gem type, calculate price - player's gems
            card_price = np.array(card.price)
            # Add offset of 10 to make all values positive
            #diff = card_price - player_gems + 10  #temp
            diff = card_price - player_gems

            diff = np.clip(diff, 0, HIGH_VALUE)  # Clip values to valid range
            diffs.extend(diff)
    
    return np.array(diffs, dtype=np.int32)

def get_high_level_price_sum_obs(board):
    """Sum of prices for level 2 and 3 cards per gem type"""
    total_price = np.zeros(5, dtype=np.int32)
    
    # Sum prices for level 2 and 3 cards
    for card in board.flatten_table_cards:
        if card is not None and card.level in [2, 3]:
            total_price += np.array(card.price)
    
    # Clip values to valid range
    total_price = np.clip(total_price, 0, HIGH_VALUE)
    
    return total_price
