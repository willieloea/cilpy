"""
This file implements the moving peaks function generator for minimization
problems.
"""

from typing import List, Callable

import math

class Peak:
    """
    A peak for the moving peaks benchmark (MPB) function generator
    """
    def __init__(self, position: List[float], depth: float, width: float):
        """
        Peak environment parameters, $e_i$, maintain the properties of a peak.
        """
        # self.minDepth: float = -10.0    # lower bound value for peak depth
        # self.maxDepth: float = 90.0     # upper bound value for peak depth
        # self.minWidth: float = 0.0      # lower bound value for peak width
        # self.maxWidth: float = 100.0    # upper bound value for peak width
        # self.domain = None              # bounds of problem domain
        self.position: List[float] = position   # peak position
        self.depth: float = depth               # peak depth
        self.width: float = width               # peak width
        self.shift_vec: List[float] = [0.0] * len(position)
                                        # shift_vec influences peak movement
                                        # during environment change 
    
    def evaluate(self, x: List[float]) -> float:
        """
        Evaluates the peak's value at a given coordinate x.
        """
        sum_sq_diff = sum((xi - vi)**2 for xi, vi in zip(x, self.position))
        return self.depth + self.width * math.sqrt(sum_sq_diff)

class MPB:
    def __init__(self):
        self.ceiling: float = 10.0
        self.peak_funcs: List[Callable[
            [List[float], List[float], float, float],
            float
        ]] = []

    def objective(self, x: List[float], t: int) -> float:
        return max(self.ceiling, self.peak_funcs(x, t))
