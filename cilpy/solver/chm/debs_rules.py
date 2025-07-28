# cilpy/solver/cht/debs_rules.py

from typing import Tuple, List
from . import ConstraintHandler
from ...problem import Problem

Fitness = Tuple[float, float]

class DebsRules(ConstraintHandler[List[float], Fitness]):
    """
    Implements Deb's feasibility rules for constraint handling.

    A solution's fitness is represented by a tuple: (total_violation, objective_value).
    Comparison logic is as follows (assuming minimization):
    1. A feasible solution is always better than an infeasible one.
    2. Between two infeasible solutions, the one with lower violation is better.
    3. Between two feasible solutions, the one with lower objective value is better.

    Python's native tuple comparison `(v_a, o_a) < (v_b, o_b)` handles this
    logic perfectly.
    """
    def __init__(self, problem: Problem[List[float], float]):
        super().__init__(problem)
        # Assuming single-objective for this classic implementation
        if len(self.objective_functions) > 1:
            raise ValueError("DebsRules currently supports single-objective problems.")
        self.objective = self.objective_functions[0]

    def evaluate(self, solution: List[float]) -> Fitness:
        """Returns (total_violation, objective_value)."""
        violation = self._calculate_total_violation(solution)
        objective = self.objective(solution)
        return (violation, objective)

    def is_better(self, fitness_a: Fitness, fitness_b: Fitness) -> bool:
        """Uses lexicographical tuple comparison."""
        return fitness_a < fitness_b