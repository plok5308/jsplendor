import random

from jsplendor.game.abs import GameComponent


class Board(GameComponent):
    def __init__(self, name, development_cards, noble_cards, coins, verbose=False, logger=None):
        super().__init__(name, development_cards, noble_cards, coins, verbose, logger)
        # parameters
        self.level1_n = 4
        self.level2_n = 4
        self.level3_n = 4

        # table information
        self.table_level1 = []
        self.table_level2 = []
        self.table_level3 = []

        self._split_development_cards()
        self._initial_lay_cards_on_table()
        self._flatten_table_cards()

    def _flatten_table_cards(self):
        self.flatten_table_cards = self.table_level1 + self.table_level2 + self.table_level3
        assert(len(self.flatten_table_cards)==12)

    def _split_development_cards(self):
        self.level1_cards = []
        self.level2_cards = []
        self.level3_cards = []

        for card in self.development_cards:
            if card.level==1:
                self.level1_cards.append(card)
            elif card.level==2:
                self.level2_cards.append(card)
            elif card.level==3:
                self.level3_cards.append(card)
            else:
                raise ValueError
            
    def _initial_lay_cards_on_table(self):
        # shuffle
        random.shuffle(self.level1_cards)
        random.shuffle(self.level2_cards)
        random.shuffle(self.level3_cards)

        for i in range(self.level1_n):
            _ = self.lay_level_card_on_table(level=1)
        for i in range(self.level2_n):
            _ = self.lay_level_card_on_table(level=2)
        for i in range(self.level3_n):
            _ = self.lay_level_card_on_table(level=3)

    def lay_level_card_on_table(self, level):
        if level==1:
            total_cards = self.level1_cards
            table_cards = self.table_level1
        elif level==2:
            total_cards = self.level2_cards
            table_cards = self.table_level2
        elif level==3:
            total_cards = self.level3_cards
            table_cards = self.table_level3
        else:
            raise ValueError("check level")
        if len(total_cards) > 0:
            table_cards.append(total_cards.pop())
            return 0
        else:
            table_cards.append(None)
            if self.verbose:
                self.logger.info(f'There is no more level{level} cards.')
            return -1

    def update_table_development_card(self, card):
        for table_card in self.table_level1:
            if table_card == card:
                self.table_level1.remove(table_card)
                self.lay_level_card_on_table(level=1)

        for table_card in self.table_level2:
            if table_card == card:
                self.table_level2.remove(table_card)
                self.lay_level_card_on_table(level=2)

        for table_card in self.table_level3:
            if table_card == card:
                self.table_level3.remove(table_card)
                self.lay_level_card_on_table(level=3)

        self._flatten_table_cards()
            
    def print_status(self, object_name=None):
        if not self.verbose:
            return