# cilpy/solver/cht/penalty_method.py

from typing import List
from . import ConstraintHandler
from ...problem import Problem

class StaticPenalty(ConstraintHandler[List[float], float]):
    """
    Implements a static penalty function.

    Fitness = f(x) + P * total_violation(x), where P is a fixed penalty coefficient.
    The goal is to minimize this combined value.
    """
    def __init__(self, problem: Problem[List[float]], penalty_coefficient: float = 1e6):
        super().__init__(problem)
        if len(self.objective_functions) > 1:
            raise ValueError("StaticPenalty currently supports single-objective problems.")
        self.objective = self.objective_functions[0]
        self.penalty_coefficient = penalty_coefficient

    def evaluate(self, solution: List[float]) -> float:
        """Returns the penalized objective value."""
        violation = self._calculate_total_violation(solution)
        objective = self.objective(solution)
        return objective + self.penalty_coefficient * violation

    def is_better(self, fitness_a: float, fitness_b: float) -> bool:
        """Compares the two penalized float values."""
        return fitness_a < fitness_b