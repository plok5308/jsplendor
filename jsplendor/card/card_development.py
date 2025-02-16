from jsplendor.card.development_list import l1_list, l2_list, l3_list


class Card:
    def __init__(self, name, level, victory_point, gem_color, price):
        self.name = name
        self.level = level
        self.victory_point = victory_point
        self.gem_color = gem_color
        self.price = price
        self.logger = None  # Initialize logger as None

    def set_logger(self, logger):
        self.logger = logger

    def __repr__(self):
        return self.name

    def print_info(self):
        if self.logger:
            self.logger.info(f'level: {self.level}')
            self.logger.info(f'victory point: {self.victory_point}')
            self.logger.info(f'gem color: {self.gem_color}')
            self.logger.info(f'price: {self.price}')
        return {
            'level': self.level,
            'victory_point': self.victory_point,
            'gem_color': self.gem_color,
            'price': self.price
        }

def get_all_development_cards():
    card_list = []
    card_list.extend(get_level_development_cards(1))
    card_list.extend(get_level_development_cards(2))
    card_list.extend(get_level_development_cards(3))

    return card_list

def get_level_development_cards(level):
    if level==1:
        data_list = l1_list
    elif level==2:
        data_list = l2_list
    elif level==3:
        data_list = l3_list
    else:
        raise ValueError("Level should be range of [1, 3].")
    
    card_list = []
    for data in data_list:
        card = Card(
            name=data[0],
            level=data[1],
            victory_point=data[2],
            gem_color=data[3],
            price=data[4])
        card_list.append(card)

    return card_list