from enum import Enum

class Element(Enum):
    WHITE=0
    BLUE=1
    GREEN=2
    RED=3
    BLACK=4
    GOLD=5

def get_verbose_dict(default:bool = False):
    verbose_dict = dict()
    verbose_dict['env'] = default
    verbose_dict['game'] = default
    verbose_dict['board'] = default
    verbose_dict['player'] = default

    return verbose_dict