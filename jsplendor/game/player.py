import numpy as np

from jsplendor.game.abs import GameComponent
from jsplendor.utils import Element
from jsplendor.game.utils import adjust_price, get_coin_comb
from jsplendor.coin import sum_coins


# TODO:
# drop coin if player has over ten coins. 
# drop rule:
# sum elements of level1 and level2 of development cards on the table.
# drop the lowest element token.

class Player(GameComponent):
    def __init__(self, name, development_cards, noble_cards, coins, verbose=False, logger=None):
        super().__init__(name, development_cards, noble_cards, coins, verbose, logger)
        self.n_coin_action = 10
        self.n_buy_action = 12
        self.num_actions = self.n_coin_action + self.n_buy_action
        self._update_score()

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

    def do_action(self, board, action):
        over_coin_count = 0
        get_card = False
        noble_visit = False
        
        if action < self.n_coin_action:
            self.get_coins(board, action)
            over_coin_count = self.drop_over_coins(board)
        else:
            get_card = self.buy_development_card(board, action-self.n_coin_action)
            if get_card:
                noble_visit = self.check_and_get_nobles(board)

        return self.sum_victory_point, over_coin_count, get_card, noble_visit

    def get_all_possible_actions(self, board):
        actions = np.zeros(self.num_actions)

        for i in range(self.n_coin_action):
            actions[i] = 1

        for i in range(self.n_coin_action, self.num_actions):
            card_position = i - self.n_coin_action
            if self.is_possible_to_buy(board, card_position):
                actions[i] = 1

        actions = actions.astype(np.bool_)
        return actions

    def get_coins(self, board, x):
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

        return count

    def drop_unnecessary_coin(self, board):
        price_sum = np.zeros(5, dtype=int)

        table_cards = board.table_level1
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
        board.coins[color] += 1
        self.coins[color] -= 1
        if self.verbose:
            self.logger.info(f'{self.name} drop a {color} coin.')

    def is_possible_to_buy(self, board, card_position):
        card = board.flatten_table_cards[card_position]

        if card is None:
            pass
            is_possible = False
        else:
            price = card.price
            price = adjust_price(price, self.sum_development_card_gem)

            is_possible = True
            for key, value in self.coins.items():
                if key=="GOLD":
                    pass
                else: 
                    if price[Element[key].value] > value:
                        is_possible = False
                        break

        return is_possible
    
    def buy_development_card(self, board, card_position):
        get_card = False
        assert(card_position>=0 and card_position<=self.n_buy_action)

        if self.is_possible_to_buy(board, card_position):
            card = board.flatten_table_cards[card_position]
            price = card.price
            price = adjust_price(price, self.sum_development_card_gem)
            
            # Log the card purchase with detailed information
            if self.verbose:
                price_info = []
                for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], price):
                    if amount > 0:
                        price_info.append(f"{color}: {amount}")
                price_str = ", ".join(price_info)
                
                self.logger.info(f'{self.name} bought card {card.name}:')
                self.logger.info(f'  Level: {card.level}')
                self.logger.info(f'  VP: {card.victory_point}')
                self.logger.info(f'  Gem Color: {card.gem_color}')
                self.logger.info(f'  Price: {price_str}')
            
            for key in self.coins.keys():
                if key=="GOLD":
                    pass
                else:
                    self.coins[key] -= int(price[Element[key].value])
                    board.coins[key] += int(price[Element[key].value])

            self.development_cards.append(card)
            board.update_table_development_card(card)
            
            self._update_score()
            
            if self.verbose:
                self.logger.info(f'VP: {self.sum_victory_point}')
            get_card = True

        else:
            if self.verbose:
                self.logger.info('Not enough tokens.')

        return get_card

    def check_and_get_nobles(self, board):
        """Check and acquire any available noble cards."""
        had_noble_visit = False
        for card in board.noble_cards[:]:  # Use slice copy to avoid modifying during iteration
            if self.is_get_possible_noble_card(card):
                self.noble_cards.append(card)
                board.noble_cards.remove(card)
                if self.verbose:
                    self.logger.info(f'Get {card} card.')
                board.noble_cards.append(None)
                had_noble_visit = True
        return had_noble_visit

    def update_noble_cards(self, board):
        """Deprecated - use check_and_get_nobles instead"""
        noble_visit = self.check_and_get_nobles(board)
        return noble_visit

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
            print("WARNING: Duplicate cards found!")
            from collections import Counter
            counts = Counter(card_names)
            for name, count in counts.items():
                if count > 1:
                    print(f"Card {name} appears {count} times")

