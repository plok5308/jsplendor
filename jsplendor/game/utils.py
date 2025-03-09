import numpy as np
from jsplendor.utils.config import get_coin_comb

def adjust_price(price, gem_status):
    """Adjust price based on gem status"""
    adjusted_price = np.array(price)
    adjusted_price = np.maximum(adjusted_price - gem_status, 0)
    return adjusted_price