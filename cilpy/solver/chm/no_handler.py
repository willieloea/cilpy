# cilpy/solver/cht/no_handler.py

from typing import TypeVar, Tuple, List
from . import ConstraintHandler
from ...problem import Problem

SolutionType = TypeVar('SolutionType')

Fitness = Tuple[float, float]

class NoHandler(ConstraintHandler[List[float], Fitness]):
    """
    Abstract interface for a Constraint Handling Mechanism (CHM).

    This interface defines a strategy for evaluating and comparing solutions
    in the context of a constrained optimization problem. A solver can be
    configured with a specific `ConstraintHandler` to dictate how it
    balances objective minimization with constraint satisfaction.
    """

    def __init__(self, problem: Problem[List[float]]):
        """Initializes the CHT with the problem it will handle."""
        super().__init__(problem)
        self.objective = self.objective_functions[0]

    def _calculate_total_violation(self, solution: List[float]) -> float:
        """
        A helper method to calculate the sum of all constraint violations.
        This is a common operation needed by most CHTs.

        Returns:
            float: The sum of violations. A value of 0.0 indicates a
                   feasible solution.
        """
        total_violation = 0.0
        # Inequality constraints are of the form g(x) <= 0
        for g in self.inequality_constraints:
            total_violation += max(0, g(solution))
        # Equality constraints are of the form h(x) = 0
        for h in self.equality_constraints:
            total_violation += abs(h(solution))
        return total_violation

    def evaluate(self, solution: List[float]) -> Fitness:
        """
        Evaluates a solution and returns its fitness representation.

        The returned `FitnessType` is used for comparison. For example, a
        penalty method might return a single float, while feasibility rules
        might return a tuple of (violation, objective_value).

        Args:
            solution (SolutionType): The solution to evaluate.

        Returns:
            FitnessType: The fitness representation of the solution.
        """
        violation = self._calculate_total_violation(solution)
        objective = self.objective(solution)
        return (violation, objective)

    def is_better(self, fitness_a: Fitness, fitness_b: Fitness) -> bool:
        """
        Compares two fitness values to determine which is better.

        This method encapsulates the core logic of the CHT (e.g., applying
        Deb's rules, comparing penalized values). Assumes minimization.

        Args:
            fitness_a: The fitness of the first solution.
            fitness_b: The fitness of the second solution.

        Returns:
            bool: True if `fitness_a` is strictly better than `fitness_b`.
        """
        return fitness_a < fitness_b

    def repair(self, solution: SolutionType) -> SolutionType:
        """
        Optionally repairs an infeasible solution to make it feasible.

        By default, this method does nothing. Subclasses that implement
        repair strategies should override this method.

        Args:
            solution (SolutionType): The solution to potentially repair.

        Returns:
            SolutionType: The repaired (or original) solution.
        """
        return solution