"""
This file implements the moving peaks function generator for maximization
problems.
"""
from typing import List, Callable

import math

class Peak:
    """
    A peak for the moving peaks benchmark (MPB) function generator
    """
    def __init__(self, position: List[float], height: float, width: float):
        """
        Peak environment parameters, $e_i$, maintain the properties of a peak.
        """
        # self.minHeight: float = 0.0     # lower bound value for peak height
        # self.maxHeight: float = 100.0   # upper bound value for peak height
        # self.minWidth: float = 0.0      # lower bound value for peak width
        # self.maxWidth: float = 100.0    # upper bound value for peak width
        # self.domain = None              # bounds of problem domain
        self.position: List[float] = position   # peak position
        self.height: float = height             # peak height
        self.width: float = width               # peak width
        self.shift_vec: List[float] = [0.0] * len(position)
                                        # shift_vec influences peak movement
                                        # during environment change 
    
    def evaluate(self, x: List[float]) -> float:
        """
        Evaluates the peak's value at a given coordinate x.
        """
        sum_sq_diff = sum((xi - vi)**2 for xi, vi in zip(x, self.position))
        return self.height - self.width * math.sqrt(sum_sq_diff)

class MPB:
    def __init__(self):
        self.basis: float = 0.0
        self.peak_funcs: List[Callable[
            [List[float], List[float], float, float],
            float
        ]] = []

    def objective(self, x: List[float], t: int) -> float:
        return max(self.basis, self.peak_funcs(x, t))
