import random
import numpy as np

from jsplendor.game import Board
from jsplendor.game import Player
from jsplendor.card import get_all_development_cards, get_three_noble_cards
from jsplendor.coin import get_full_coin_for_board, get_empty_coin
from jsplendor.utils import get_verbose_dict


class Game:
    def __init__(self, verbose_dict=None, gui=None):
        self.gui = gui  # Store reference to GUI
        if verbose_dict is None:
            self.verbose_dict = get_verbose_dict()
        else:
            self.verbose_dict = verbose_dict

        self.verbose = self.verbose_dict['game']
        self.reset()

    def reset(self):
        all_development_cards = get_all_development_cards()
        selected_noble_cards = get_three_noble_cards()
        board_coins = get_full_coin_for_board()
        self.board = Board(
                    name="board",
                    development_cards=all_development_cards,
                    noble_cards=selected_noble_cards,
                    coins=board_coins,
                    verbose=self.verbose_dict['board'])
        
        player1_coins = get_empty_coin()
        self.player1 = Player(
                        name="player1",
                        development_cards=[],
                        noble_cards=[],
                        coins=player1_coins,
                        verbose=self.verbose_dict['player'])
        
        self.component_list = [self.board, self.player1]

        if self.verbose:
            print("Initial game status")
            self.print_status()

        self.step = 0
        self.check_all_coins()
        self.check_all_cards()

    def print_status(self):
        if self.verbose:
            if self.verbose_dict['board']:
                self.board.print_status(object_name="board")
            if self.verbose_dict['player']:
                self.player1.print_status(object_name="player1")
        else:
            pass

    def get_random_action_datas(self):  # for collecting data
        actions_bool = self.player1.get_all_possible_actions(self.board)
        possible_actions = np.where(actions_bool==1)[0].tolist()
        action = random.choice(possible_actions)
        action_result = self.run_with_action(action)

        return action, action_result, actions_bool

    def run_with_action(self, action):
        self.step += 1
        action_result = dict()
        action_result['victory_point'] = 0
        action_result['over_coin_count'] = 0
        action_result['is_skip'] = False
        action_result['is_get_card'] = False
        action_result['is_noble_visit'] = False
        action_result['step'] = self.step

        actions_bool = self.player1.get_all_possible_actions(self.board)

        if actions_bool[action]:
            victory_point, over_coin_count, get_card, noble_visit = self.player1.do_action(self.board, action)
            action_result['victory_point'] = victory_point
            action_result['over_coin_count'] = over_coin_count
            action_result['is_get_card'] = get_card
            action_result['is_noble_visit'] = noble_visit
            
            # Add log messages
            if self.gui:
                if get_card:
                    self.gui.add_log_message(f"Step {self.step}: Got a card")
                if over_coin_count > 0:
                    self.gui.add_log_message(f"Step {self.step}: Dropped {over_coin_count} coins")
                if noble_visit:
                    self.gui.add_log_message(f"Step {self.step}: Attracted a noble")
        else:  # invalid action
            action_result['is_skip'] = True
            if self.gui:
                self.gui.add_log_message(f"Step {self.step}: Invalid action")

        return action_result

    def check_all_coins(self):
        sum_white = 0
        sum_blue = 0
        sum_green = 0
        sum_red = 0
        sum_black = 0
        sum_gold = 0

        for component in self.component_list:
            sum_white += component.coins["WHITE"]
            sum_blue += component.coins["BLUE"]
            sum_green += component.coins["GREEN"]
            sum_red += component.coins["RED"]
            sum_black += component.coins["BLACK"]
            sum_gold += component.coins["GOLD"]

        assert(sum_white==4)
        assert(sum_blue==4)
        assert(sum_green==4)
        assert(sum_red==4)
        assert(sum_black==4)
        assert(sum_gold==5)

    def check_all_cards(self):
        sum_developement_cards = 0
        sum_noble = 0

        for component in self.component_list:
            if isinstance(component, Board):
                sum_developement_cards += len(component.level1_cards) + len(component.table_level1)
                sum_developement_cards += len(component.level2_cards) + len(component.table_level2)
                sum_developement_cards += len(component.level3_cards) + len(component.table_level3)
            else:  # player
                sum_developement_cards += len(component.development_cards)

            sum_noble += len(component.noble_cards)

        assert(sum_developement_cards == 40+30+20), sum_developement_cards  # level1 + level2 + level3
        assert(sum_noble == 3)
