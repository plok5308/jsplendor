from enum import Enum
import numpy as np

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

def get_coin_comb(x):
    """Get coin combination for a given action
    
    Args:
        x (int): Action index (must be between 0 and 9)
    Returns:
        np.array: Array of 3 coin indices
    """
    assert 0 <= x < 10, f"Action index must be between 0 and 9, got {x}"
    
    coin_comb = np.array([
        [0,1,2], [0,1,3], [0,1,4], [0,2,3], [0,2,4], [0,3,4],
        [1,2,3], [1,2,4], [1,3,4], [2,3,4]
    ])
    return coin_comb[x] 