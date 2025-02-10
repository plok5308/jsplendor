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
    def __init__(self, name, development_cards, noble_cards, coins, verbose=False):
        super().__init__(name, development_cards, noble_cards, coins, verbose)
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
        for ids in candidated_ids:
            color = Element(ids).name
            if self.is_possible_to_get_coin(board, color):
                self.get_a_coin(board, color)
    
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
                print('{} get a {} coin.'.format(self.name, color))
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
                print('{} has over coins.'.format(self.name))
                print(self.coins)
            self.drop_unnecessary_coin(board)
            sum_v = sum_coins(self.coins)
            count += 1

        return count

    def drop_unnecessary_coin(self, board):
        price_sum = np.zeros(5, dtype=int)

        #table_cards = board.table_level1 + board.table_level2
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
                break

    def drop_a_coin(self, board, color):
        board.coins[color] += 1
        self.coins[color] -= 1
        if self.verbose:
            print('{} drop a {} coin.'.format(self.name, color))

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
            
            for key in self.coins.keys():
                if key=="GOLD":
                    pass
                else:
                    self.coins[key] -= int(price[Element[key].value])
                    board.coins[key] += int(price[Element[key].value])

            self.development_cards.append(card)
            self._update_score()
            
            if self.verbose:
                print('{} get a {} card.'.format(self.name, card))

            board.update_table_development_card(card)

            self.update_noble_cards(board)
            self._update_score()
            get_card = True
            if self.verbose:
                print('VP: {}.'.format(self.sum_victory_point))

        else:
            if self.verbose:
                print('Not enough tokens.')

        return get_card

    def check_and_get_nobles(self, board):
        """Check and acquire any available noble cards."""
        had_noble_visit = False
        for card in board.noble_cards[:]:  # Use slice copy to avoid modifying during iteration
            if self.is_get_possible_noble_card(card):
                self.noble_cards.append(card)
                board.noble_cards.remove(card)
                if self.verbose:
                    print('Get {} card.'.format(card))
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
        if object_name is not None:
            print("[{}]".format(object_name))

        print("Development cards: ")
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

        print(l1_cards)
        print(l2_cards)
        print(l3_cards)
        
        print("Noble cards: ")
        print(self.noble_cards)
        print("Coin status: ")
        print(self.coins)    
        print("Victory point: {}".format(self.sum_victory_point))
        print("")

