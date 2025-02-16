import random
# price array - [(w)hite, bl(u)e, (g)reen, (r)ed, blac(k)]
# price array - [w, u, g, r, k]

# name / VP / price 
noble_list = [
    ('noble1', 3, [0, 0, 4, 4, 0]),
    ('noble2', 3, [3, 0, 0, 3, 3]),
    ('noble3', 3, [4, 4, 0, 0, 0]),
    ('noble4', 3, [4, 0, 0, 0, 4]),
    ('noble5', 3, [0, 4, 4, 0, 0]),
    ('noble6', 3, [0, 3, 3, 3, 0]),
    ('noble7', 3, [3, 3, 3, 0, 0]),
    ('noble8', 3, [0, 0, 0, 4, 4]),
    ('noble9', 3, [3, 3, 0, 0, 3]),
    ('noble10', 3, [0, 0, 3, 3, 3]),
]


class NobleCard:
    def __init__(self, name, victory_point, price):
        self.name = name
        self.victory_point = victory_point
        self.price = price
        self.logger = None  # Initialize logger as None

    def set_logger(self, logger):
        self.logger = logger

    def __repr__(self):
        return self.name


def get_noble_cards():
    noble_cards = []
    for data in noble_list:
        card = NobleCard(
            name=data[0],
            victory_point=data[1],
            price=data[2])
        noble_cards.append(card)
        
    return noble_cards

def get_three_noble_cards():
    noble_cards = get_noble_cards()
    selected_noble_cards = random.sample(noble_cards, 3)
    return selected_noble_cards


