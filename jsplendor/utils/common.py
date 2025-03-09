from enum import Enum

class Element(Enum):
    WHITE=0
    BLUE=1
    GREEN=2
    RED=3
    BLACK=4
    GOLD=5

def get_verbose_dict(verbose=False):
    """Get dictionary of verbose settings for different components"""
    verbose_dict = {
        'env': verbose,
        'game': verbose,
        'board': verbose,
        'player': verbose
    }
    return verbose_dict