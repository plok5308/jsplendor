import random
import numpy as np

from jsplendor.game import Board
from jsplendor.game import Player
from jsplendor.card import get_all_development_cards, get_three_noble_cards
from jsplendor.coin import get_full_coin_for_board, get_empty_coin
from jsplendor.utils.config import get_verbose_dict
from jsplendor.utils import TestLogger


class Game:
    def __init__(self, verbose_dict=None, gui=None):
        self.gui = gui
        if verbose_dict is None:
            self.verbose_dict = get_verbose_dict()
        else:
            self.verbose_dict = verbose_dict

        self.verbose = self.verbose_dict['game']
        if self.verbose:
            self.logger = TestLogger.get_logger()
        
        # Initialize players list
        self.players = []
        self.reset()

    def reset(self):
        all_development_cards = get_all_development_cards()
        selected_noble_cards = get_three_noble_cards()
        
        # Initialize board
        self.board = Board(
                    name="board",
                    development_cards=all_development_cards,
                    noble_cards=selected_noble_cards,
                    coins=get_full_coin_for_board(),
                    verbose=self.verbose_dict['board'],
                    logger=TestLogger.get_logger() if self.verbose_dict['board'] else None)
        
        # Store current players' names
        player_names = [p.name for p in self.players]
        
        # Clear and reinitialize players
        self.players = []
        
        # Add at least one player
        if not player_names:
            player_names = ["player1"]
        
        # Recreate all players
        for name in player_names:
            self.add_player(name)

        if self.verbose:
            self.logger.info("Initial game status")
            self.print_status()

        self.check_all_coins()
        self.check_all_cards()

    def add_player(self, name):
        """Add a new player to the game"""
        # Create logger first if verbose is enabled
        logger = None
        if self.verbose_dict['player']:
            logger = TestLogger.get_logger('logs/game')  # Use singleton logger
        
        player = Player(
            name=name,
            development_cards=[],
            noble_cards=[],
            coins=get_empty_coin(),
            verbose=self.verbose_dict['player'],
            logger=logger
        )
        self.players.append(player)
        return player

    def print_status(self):
        if not self.verbose:
            return
            
        self.logger.info("Game status")
        if self.verbose_dict['board']:
            self.board.print_status(object_name="board")
        
        for player in self.players:
            if self.verbose_dict['player']:
                player.print_status(object_name=player.name)

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

        if action >= 15 and action < 20:
            action_result['buy_l1_card'] = True
        else:
            action_result['buy_l1_card'] = False

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
                if action >= 10 and action < 15:  # Log double coin collection
                    color = Element(action - 15).name
                    self.gui.add_log_message(f"Step {self.step}: Got two {color} coins")
        else:  # invalid action
            action_result['is_skip'] = True
            if self.gui:
                self.gui.add_log_message(f"Step {self.step}: Invalid action")

        return action_result

    def check_all_coins(self):
        """Verify total coins across all components"""
        sum_white = sum_blue = sum_green = sum_red = sum_black = sum_gold = 0

        # Count board coins
        sum_white += self.board.coins["WHITE"]
        sum_blue += self.board.coins["BLUE"]
        sum_green += self.board.coins["GREEN"]
        sum_red += self.board.coins["RED"]
        sum_black += self.board.coins["BLACK"]
        sum_gold += self.board.coins["GOLD"]

        # Count player coins
        for player in self.players:
            sum_white += player.coins["WHITE"]
            sum_blue += player.coins["BLUE"]
            sum_green += player.coins["GREEN"]
            sum_red += player.coins["RED"]
            sum_black += player.coins["BLACK"]
            sum_gold += player.coins["GOLD"]

        assert sum_white == 4, f"White coins don't sum to 4: {sum_white}"
        assert sum_blue == 4, f"Blue coins don't sum to 4: {sum_blue}"
        assert sum_green == 4, f"Green coins don't sum to 4: {sum_green}"
        assert sum_red == 4, f"Red coins don't sum to 4: {sum_red}"
        assert sum_black == 4, f"Black coins don't sum to 4: {sum_black}"
        assert sum_gold == 5, f"Gold coins don't sum to 5: {sum_gold}"

    def check_all_cards(self):
        """Verify total cards across all components"""
        sum_development_cards = 0
        sum_noble = 0

        # Count board cards
        sum_development_cards += (len(self.board.level1_cards) + len(self.board.table_level1) +
                                len(self.board.level2_cards) + len(self.board.table_level2) +
                                len(self.board.level3_cards) + len(self.board.table_level3))
        sum_noble += len(self.board.noble_cards)

        # Count player cards
        for player in self.players:
            sum_development_cards += len(player.development_cards)
            sum_noble += len(player.noble_cards)

        assert sum_development_cards == 90, f"Development cards don't sum to 90: {sum_development_cards}"
        assert sum_noble == 3, f"Noble cards don't sum to 3: {sum_noble}"

    def do_action(self, player_idx, action):
        """Execute an action for the specified player"""
        result = self.players[player_idx].do_action(self.board, action)
        return result