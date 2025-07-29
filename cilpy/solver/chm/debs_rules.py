# cilpy/solver/chm/debs_rules.py

from typing import Tuple, Any, TypeVar, Callable

from . import ConstraintHandler
from ...problem import Problem

# CHANGED: Define a generic SolutionType. This allows DebsRules to work with
# np.ndarray, List[float], or any other solution representation.
SolutionType = TypeVar("SolutionType")

# This fitness type is specific to DebsRules: (total_violation, objective_value)
Fitness = Tuple[float, float]


# CHANGED: The class now uses the generic SolutionType.
class DebsRules(ConstraintHandler[SolutionType, Fitness]):
    """
    Implements Deb's feasibility rules for constraint handling.

    A solution's fitness is represented by a tuple: (total_violation, objective_value).
    This class is generic and works with any solution type (e.g., np.ndarray, List[float]).

    Comparison logic is as follows (assuming minimization):
    1. A feasible solution is always better than an infeasible one.
    2. Between two infeasible solutions, the one with lower violation is better.
    3. Between two feasible solutions, the one with lower objective value is better.

    Python's native tuple comparison `(v_a, o_a) < (v_b, o_b)` handles this logic.
    """

    objective: Callable[[SolutionType], float]
    # We use Problem[SolutionType, Any] because DebsRules doesn't care about the
    # problem's native fitness type; it creates its own.
    def __init__(self, problem: Problem[SolutionType, Any]):
        super().__init__(problem)
        # Assuming single-objective for this classic implementation
        if len(self.objective_functions) > 1:
            raise ValueError("DebsRules currently supports single-objective problems.")
        self.objective = self.objective_functions[0]

    def evaluate(self, solution: SolutionType) -> Fitness:
        """Returns (total_violation, objective_value)."""
        violation = self._calculate_total_violation(solution)
        objective = self.objective(solution)
        # The objective value is cast to float for consistency in the fitness tuple.
        return (violation, float(objective))

    def is_better(self, fitness_a: Fitness, fitness_b: Fitness) -> bool:
        """Uses lexicographical tuple comparison."""
        return fitness_a < fitness_b