import numpy as np

from jsplendor.game.abs import GameComponent
from jsplendor.utils import Element
from jsplendor.game.utils import adjust_price, get_coin_comb
from jsplendor.coin import sum_coins
from jsplendor.utils.logger import TestLogger


# TODO:
# drop coin if player has over ten coins. 
# drop rule:
# sum elements of level1 and level2 of development cards on the table.
# drop the lowest element token.

class Player(GameComponent):
    def __init__(self, name, development_cards, noble_cards, coins, verbose=False, logger=None):
        super().__init__(name, development_cards, noble_cards, coins, verbose, logger)
        
        # Create logger if not provided but verbose is True
        if verbose and logger is None:
            self.logger = TestLogger.get_logger('logs/game')  # Use singleton logger
        else:
            self.logger = logger
        
        # Action space parameters
        self.n_coin_action = 15  # 10 for three different coins + 5 for two same coins
        self.n_buy_action = 12   # 4 cards x 3 levels
        self.n_buy_reserved_card_action = 3  # Up to 3 reserved cards
        self.n_reserve_action = 12  # 4 cards x 3 levels that can be reserved
        self.n_pass_action = 1
        
        # Total number of actions
        self.num_actions = (self.n_coin_action + 
                          self.n_buy_action + 
                          self.n_buy_reserved_card_action + 
                          self.n_reserve_action +
                          self.n_pass_action)  # Total 43 actions
        
        # Initialize other attributes
        self.development_cards = []
        self.noble_cards = []
        self.reserved_cards = []
        self.sum_victory_point = 0
        self.step = 0
        self._update_score()
        self.reserve_masking = False

        # Initialize coins with regular integers
        self.coins = {color: int(amount) for color, amount in self.coins.items()}

    def set_reserve_masking(self, reserve_masking):
        self.reserve_masking = reserve_masking

    def _update_score(self):
        sum_victory_point = 0
        sum_development_card_gem = np.zeros(5, dtype=int)
        for card in self.development_cards:
            sum_victory_point += card.victory_point
            gem_idx = Element[card.gem_color].value
            sum_development_card_gem[gem_idx] += 1

        for card in self.noble_cards:
            sum_victory_point += card.victory_point

        self.sum_victory_point = sum_victory_point
        self.sum_development_card_gem = sum_development_card_gem

    def _get_action_type_and_index(self, action):
        if action < self.n_coin_action:
            return "coin", action
        elif action < self.n_coin_action + self.n_buy_action:
            return "buy", action - self.n_coin_action
        elif action < self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action:
            return "buy_reserved_card", action - self.n_coin_action - self.n_buy_action
        elif action < self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action + self.n_reserve_action:
            return "reserve", action - self.n_coin_action - self.n_buy_action - self.n_buy_reserved_card_action
        elif action == self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action + self.n_reserve_action:
            return "pass", 0
        else:
            raise ValueError(f"Invalid action: {action}")

    def do_action(self, board, action):
        self.step += 1  # Increment step when player takes action
        over_coin_count = 0
        get_card = False
        noble_visit = False
        spent_coins = None  # Initialize spent_coins variable

        if self.verbose:
            self.logger.info(f"Action: {action}")

        action_type, action_index = self._get_action_type_and_index(action)
        
        # Logging at start of turn with clear separator
        if self.verbose:
            self.logger.info("\n" + "="*50)
            self.logger.info(f"Turn {self.step} - {self.name}")
            self.logger.info("-"*25)
            
            # Board State
            self.logger.info("Board State:")
            self.logger.info(f"Coins: {board.coins}")
            self.logger.info("Cards:")
            self.logger.info(f"Level 3: {[card.name if card else 'Empty' for card in board.table_level3]}")
            self.logger.info(f"Level 2: {[card.name if card else 'Empty' for card in board.table_level2]}")
            self.logger.info(f"Level 1: {[card.name if card else 'Empty' for card in board.table_level1]}")
            self.logger.info(f"Nobles: {[card.name if card else 'Empty' for card in board.noble_cards]}")
            self.logger.info("-"*25)
            
            # Player State
            self.logger.info("Player State:")
            self.logger.info(f"Coins: {self.coins}")
            self.logger.info(f"Development cards: {[card.name for card in self.development_cards]}")
            self.logger.info(f"Reserved cards: {[card.name for card in self.reserved_cards]}")
            self.logger.info(f"Victory Points: {self.sum_victory_point}")
            self.logger.info("-"*25)
            self.logger.info(f"Action: {action_type.upper()} (index: {action_index})")
        
        # Execute action
        if action_type == "coin":
            self.get_coins(board, action_index)
            over_coin_count = self.drop_over_coins(board)
            self._ensure_regular_integers()  # Ensure regular integers after coin actions
            # Check coins after coin actions
            if hasattr(board, 'game'):
                board.game.check_all_coins()

        elif action_type == "buy":
            if self.verbose:
                card = board.flatten_table_cards[action_index]
                if card:
                    self.logger.info(f"Buying card: {card.name}")
                    self.logger.info(f"Card details - Level: {card.level}, VP: {card.victory_point}, Color: {card.gem_color}")
                    self.logger.info(f"Price: {card.price}")
            
            get_card, spent_coins = self.buy_development_card_on_table(board, action_index)
            
            if get_card and self.verbose:
                self.logger.info("Purchase successful!")
            
            if get_card:
                noble_visit = self.check_and_get_a_noble(board)
                if spent_coins:
                    for color, amount in spent_coins.items():
                        board.coins[color] += int(amount)
                    if self.verbose:
                        clean_coins = {k: int(v) for k, v in spent_coins.items()}
                        self.logger.info(f"Coins returned to board: {clean_coins}")
                # Check for duplicates after buying card
                self._check_for_duplicates()
                if hasattr(board, 'game'):
                    board.game.check_all_coins()
                self._ensure_regular_integers()

        elif action_type == "buy_reserved_card":
            if self.verbose:
                if action_index < len(self.reserved_cards):
                    card = self.reserved_cards[action_index]
                    self.logger.info(f"Attempting to buy reserved card: {card.name}")
                    self.logger.info(f"Card details - Level: {card.level}, VP: {card.victory_point}, Color: {card.gem_color}")
                    self.logger.info(f"Card price: {card.price}")
                else:
                    self.logger.info(f"Invalid reserved card index {action_index}. Have {len(self.reserved_cards)} reserved cards.")
            
            get_card, spent_coins = self.buy_reserved_card(board, action_index)
            
            if get_card and self.verbose:
                self.logger.info(f"Reserved card purchase successful!")
            
            if get_card:
                noble_visit = self.check_and_get_a_noble(board)
                if spent_coins:
                    for color, amount in spent_coins.items():
                        board.coins[color] += int(amount)
                # Check for duplicates after buying reserved card
                self._check_for_duplicates()
                if hasattr(board, 'game'):
                    board.game.check_all_coins()
                self._ensure_regular_integers()

        elif action_type == "reserve":
            if self.verbose:
                card = board.flatten_table_cards[action_index]
                if card:
                    self.logger.info(f"Reserving card: {card.name}")
                    self.logger.info(f"Card details - Level: {card.level}, VP: {card.victory_point}, Color: {card.gem_color}")
            
            self.reserve_development_card(board, action_index)
            over_coin_count = self.drop_over_coins(board)
            self._ensure_regular_integers()  # Ensure regular integers after reserve
            
            if self.verbose:
                self.logger.info(f"Updated reserved cards: {[card.name for card in self.reserved_cards]}")
                if len(self.reserved_cards) >= 3:
                    self.logger.info("Warning: Reserved card limit reached (3 cards)")

        elif action_type == "pass":
            if self.verbose:
                self.logger.info("Passing turn")

        # Remove general coin check since we now check after each specific action
        self._update_score()

        self.check_player_coins()

        if self.verbose:
            self.logger.info(f"* Victory Points after action: {self.sum_victory_point}")

        return self.sum_victory_point, over_coin_count, get_card, noble_visit

    def get_all_possible_actions(self, board):
        actions = np.zeros(self.num_actions)

        # Original coin actions (0-9)
        for i in range(self.n_coin_action - 5):  # First 10 actions are for 3 different coins
            candidated_ids = get_coin_comb(i)
            available_coins = 0
            for ids in candidated_ids:
                color = Element(ids).name
                if self.is_possible_to_get_coin(board, color):
                    available_coins += 1
            if available_coins > 0:
                actions[i] = 1

        # Double coin actions (10-14)
        for i in range(5):  # Loop through 0-4 for the 5 colors
            if self.is_possible_to_get_two_coins(board, Element(i).name):
                actions[i + 10] = 1  # Map to actions 10-14

        # Buy card actions (15-26)
        for i in range(self.n_coin_action, self.n_coin_action + self.n_buy_action):
            card_position = i - self.n_coin_action
            if self.is_possible_to_buy_card_on_table(board, card_position):
                actions[i] = 1

        # Buy reserved card actions (27-29)
        for i in range(self.n_coin_action + self.n_buy_action, 
                      self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action):
            card_position = i - (self.n_coin_action + self.n_buy_action)
            if card_position < len(self.reserved_cards):
                if self.is_possible_to_buy_card(self.reserved_cards[card_position]):
                    actions[i] = 1

        # Reserve card actions (30-41)
        if self.reserve_masking:
            pass
        else:
            for i in range(self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action,
                        self.num_actions - 1):
                if len(self.reserved_cards) < 3:  # Can only reserve if less than 3 cards
                    card_position = i - (self.n_coin_action + self.n_buy_action + self.n_buy_reserved_card_action)
                    card = board.flatten_table_cards[card_position]
                    if card is not None:
                        actions[i] = 1

        if np.sum(actions) == 0:
            actions[self.num_actions - 1] = 1  # Pass action

        actions = actions.astype(np.bool_)
        
        return actions

    def get_coins(self, board, x):
        if x >= 10 and x < 15:  # get two different coins
            color = Element(x - 10).name
            if self.is_possible_to_get_two_coins(board, color):
                board.coins[color] -= 2
                self.coins[color] += 2
                if self.verbose:
                    self.logger.info(f'{self.name} got two {color} coins')
        else:  # get three different coins
            candidated_ids = get_coin_comb(x)
            collected_coins = []  # Track which coins were collected
            
            for ids in candidated_ids:
                color = Element(ids).name
                if self.is_possible_to_get_coin(board, color):
                    self.get_a_coin(board, color)
                    collected_coins.append(color)
            
            # Log the collected coins
            if collected_coins and self.verbose:
                coins_msg = "Collected coins: " + ", ".join(collected_coins)
                self.logger.info(coins_msg)

        # Convert any numpy integers to regular integers when getting coins
        for color in self.coins.keys():
            self.coins[color] = int(self.coins[color])

    def is_possible_to_get_two_coins(self, board, color):
        if board.coins[color] >= 4:
            return True
        else:
            return False

    def is_possible_to_get_coin(self, board, color):
        if board.coins[color] > 0:
            return True
        else:
            return False

    def get_a_coin(self, board, color):
        if board.coins[color] > 0:
            board.coins[color] -= 1
            self.coins[color] += 1
            if self.verbose:
                self.logger.info(f'{self.name} get a {color} coin.')
        else:
            pass

    def has_a_coin(self, color):
        if self.coins[color] > 0:
            return True
        else:
            return False

    def drop_over_coins(self, board):
        sum_v = sum_coins(self.coins)

        count = 0
        while (sum_v > 10):
            if self.verbose:
                self.logger.info(f'{self.name} has over coins.')
                self.logger.info(str(self.coins))
            self.drop_unnecessary_coin(board)
            sum_v = sum_coins(self.coins)
            count += 1

        # Ensure regular integers after dropping coins
        for color in self.coins.keys():
            self.coins[color] = int(self.coins[color])
        
        return count

    def drop_unnecessary_coin(self, board):
        price_sum = np.zeros(5, dtype=int)

        table_cards = board.table_level1 + board.table_level2 + board.table_level3
        for card in table_cards:
            if card is None:
                pass
            else:
                price_sum += np.array(card.price)
            
        idx_array = np.argsort(price_sum)
        for idx in idx_array:
            color = Element(idx).name
            if self.has_a_coin(color):
                self.drop_a_coin(board, color)
                if self.verbose:
                    self.logger.info(f'Dropped {color} coin (over 10 limit)')
                break

    def drop_a_coin(self, board, color):
        board.coins[color] = int(board.coins[color] + 1)
        self.coins[color] -= 1
        if self.verbose:
            self.logger.info(f'{self.name} drop a {color} coin.')
        # Check coins after dropping
        if hasattr(board, 'game'):
            board.game.check_all_coins()

    def is_possible_to_buy_card(self, card):
        if card is None:
            return False
            
        price = card.price
        price = adjust_price(price, self.sum_development_card_gem)
        
        # Calculate how many gold coins we'll need
        gold_needed = 0
        for key in self.coins.keys():
            if key != "GOLD":
                if price[Element[key].value] > self.coins[key]:
                    gold_needed += price[Element[key].value] - self.coins[key]
        
        # Check if we have enough gold coins
        if gold_needed > self.coins["GOLD"]:
            return False
            
        return True

    def is_possible_to_buy_card_on_table(self, board, card_position):
        card = board.flatten_table_cards[card_position]
        return self.is_possible_to_buy_card(card)
    
    def buy_development_card(self, card, board):
        original_price = card.price
        adjusted_price = adjust_price(original_price.copy(), self.sum_development_card_gem)
        gold_needed = 0
        spent_coins = {'WHITE': 0, 'BLUE': 0, 'GREEN': 0, 'RED': 0, 'BLACK': 0, 'GOLD': 0}

        # First calculate how many gold coins we need
        for key in self.coins.keys():
            if key != "GOLD":
                if self.coins[key] < adjusted_price[Element[key].value]:
                    gold_needed += adjusted_price[Element[key].value] - self.coins[key]

        # Verify we have enough gold coins
        assert gold_needed <= self.coins["GOLD"], f"Not enough gold coins. Need {gold_needed} but have {self.coins['GOLD']}"

        # Now spend the coins
        for key in self.coins.keys():
            if key != "GOLD":
                price = adjusted_price[Element[key].value]
                if price > 0:
                    # Use as many regular coins as possible
                    coins_to_use = min(self.coins[key], price)
                    self.coins[key] -= coins_to_use
                    spent_coins[key] = int(coins_to_use)  # Convert to regular int
                    # Use gold coins for the remainder if needed
                    if coins_to_use < price:
                        gold_to_use = price - coins_to_use
                        self.coins["GOLD"] -= gold_to_use
                        spent_coins["GOLD"] = int(spent_coins["GOLD"] + gold_to_use)  # Convert to regular int

        # Add card to development cards
        self.development_cards.append(card)
        # Check for duplicates after adding card
        self._check_for_duplicates()

        # Convert any numpy integers in coins to regular integers
        for color in self.coins.keys():
            self.coins[color] = int(self.coins[color])

        return spent_coins

    def buy_development_card_on_table(self, board, card_position):
        get_card = False
        spent_coins = None
        assert(card_position>=0 and card_position<=self.n_buy_action)

        if self.is_possible_to_buy_card_on_table(board, card_position):
            card = board.flatten_table_cards[card_position]
            spent_coins = self.buy_development_card(card, board)
            board.update_table_development_card(card)
            get_card = True
            # Check for duplicates after buying card
            self._check_for_duplicates()
            
            if self.verbose:
                self.logger.info(f"Coins returned to board: {spent_coins}")
        else:
            if self.verbose:
                self.logger.info('Not enough tokens.')

        return get_card, spent_coins

    def reserve_development_card(self, board, card_position):
        if len(self.reserved_cards) < 3:
            card = board.flatten_table_cards[card_position]
            self.reserved_cards.append(card)
            board.update_table_development_card(card)
            
            # Add gold coin if available
            if self.verbose:
                self.logger.info(f'Board coins before reserve: {board.coins}')
                
            if board.coins['GOLD'] > 0:
                board.coins['GOLD'] -= 1
                self.coins['GOLD'] = int(self.coins['GOLD'] + 1)
                
                # Ensure all coins are regular integers
                for color in self.coins.keys():
                    self.coins[color] = int(self.coins[color])
                
                # Check coins after gold transfer
                if hasattr(board, 'game'):
                    board.game.check_all_coins()
                if self.verbose:
                    self.logger.info(f'{self.name} received a GOLD coin for reserving.')
                    self.logger.info(f'Board coins after reserve: {board.coins}')
            else:
                if self.verbose:
                    self.logger.info('No GOLD coins available on board')
                    
        else:
            if self.verbose:
                self.logger.info('Reserved cards are full.')

    def buy_reserved_card(self, board, card_position):  # Add board parameter
        get_card = False
        spent_coins = None

        # Check if card_position is valid
        if card_position >= len(self.reserved_cards):
            if self.verbose:
                self.logger.info(f'Invalid reserved card position: {card_position}. Only have {len(self.reserved_cards)} cards.')
            return get_card, spent_coins

        card = self.reserved_cards[card_position]

        if self.is_possible_to_buy_card(card):
            spent_coins = self.buy_development_card(card, board)
            get_card = True
            # Remove card from reserved cards (card is already added to development cards in buy_development_card)
            self.reserved_cards.pop(card_position)  # Use pop instead of remove to avoid duplicate issues
        else:
            if self.verbose:
                self.logger.info('Not enough tokens.')

        return get_card, spent_coins

    def check_and_get_a_noble(self, board):
        """Check and acquire a noble card."""
        had_noble_visit = False
        for card in board.noble_cards[:]:  # Use slice copy to avoid modifying during iteration
            if self.is_get_possible_noble_card(card):
                self.noble_cards.append(card)
                board.noble_cards.remove(card)
                if self.verbose:
                    self.logger.info(f'Get {card} card.')
                board.noble_cards.append(None)
                had_noble_visit = True
                break

        return had_noble_visit

    def is_get_possible_noble_card(self, card):
        if card is None:
            is_possible = False
        else:
            price = card.price
            gem_status = self.sum_development_card_gem
            is_possible = True
            for i in range(5):
                if gem_status[i] < price[i]:
                    is_possible = False
                    break

        return is_possible

    def print_status(self, object_name=None):
        if not self.verbose:
            return
            
        if object_name is not None:
            self.logger.info(f"[{object_name}]")

        self.logger.info("Development cards: ")
        l1_cards = []
        l2_cards = []
        l3_cards = []
        for card in self.development_cards:
            if card.level==1:
                l1_cards.append(card)
            if card.level==2:
                l2_cards.append(card)
            if card.level==3:
                l3_cards.append(card)

        self.logger.info(str(l1_cards))
        self.logger.info(str(l2_cards))
        self.logger.info(str(l3_cards))
        
        self.logger.info("Noble cards: ")
        self.logger.info(str(self.noble_cards))
        self.logger.info("Coin status: ")
        self.logger.info(str(self.coins))    
        self.logger.info(f"Victory point: {self.sum_victory_point}")
        self.logger.info("")

    def _check_for_duplicates(self):
        card_names = [card.name for card in self.development_cards]
        unique_names = set(card_names)
        if len(card_names) != len(unique_names):
            if self.verbose:
                self.logger.info("WARNING: Duplicate cards found!")
                from collections import Counter
                counts = Counter(card_names)
                for name, count in counts.items():
                    if count > 1:
                        self.logger.info(f"Card {name} appears {count} times")

    def _ensure_regular_integers(self):
        """Convert any numpy integers in coins to regular integers"""
        for color in self.coins.keys():
            self.coins[color] = int(self.coins[color])

    def check_player_coins(self):
        for color in self.coins.keys():
            if self.coins[color] < 0:
                raise ValueError(f"WARNING: {self.name} has negative {color} coins: {self.coins[color]}")
        
        sum_coins = sum(self.coins.values())
        if sum_coins > 10:
            raise ValueError(f"WARNING: {self.name} has over 10 coins: {sum_coins}")
