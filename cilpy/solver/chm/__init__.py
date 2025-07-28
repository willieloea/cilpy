# cilpy/solver/chm/__init__.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, List

from ...problem import Problem

# Generic type for the solution representation (e.g., List[float])
SolutionType = TypeVar('SolutionType')

# Generic type for the fitness value produced by the CHM.
# This could be a float, a tuple, or any other comparable object.
FitnessType = TypeVar('FitnessType')


class ConstraintHandler(ABC, Generic[SolutionType, FitnessType]):
    """
    Abstract interface for a Constraint Handling Mechanism (CHM).

    This interface defines a strategy for evaluating and comparing solutions
    in the context of a constrained optimization problem. A solver can be
    configured with a specific `ConstraintHandler` to dictate how it
    balances objective minimization with constraint satisfaction.
    """

    def __init__(self, problem: Problem[SolutionType, FitnessType]):
        """Initializes the CHT with the problem it will handle."""
        self.problem = problem
        self.objective_functions = self.problem.get_objective_functions()
        self.inequality_constraints, self.equality_constraints = \
            self.problem.get_constraint_functions()

    def _calculate_total_violation(self, solution: SolutionType) -> float:
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

    @abstractmethod
    def evaluate(self, solution: SolutionType) -> FitnessType:
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
        pass

    @abstractmethod
    def is_better(self, fitness_a: FitnessType, fitness_b: FitnessType) -> bool:
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
        pass

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