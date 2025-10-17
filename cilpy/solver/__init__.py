"""
The solver module.
"""
# cilpy/solver/__init__.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, List
from ..problem import Problem

# Generic types for solutions and fitness values
SolutionType = TypeVar("SolutionType")
FitnessType = TypeVar("FitnessType")


class Solver(ABC, Generic[SolutionType, FitnessType]):
    """
    An abstract interface for a problem solver.

    All solvers in `cilpy.solver` should implement this interface. The interface
    is generic to support different solution representations (e.g., List[float],
    List[int], custom objects).

    Args:
        TODO
    """

    def __init__(self, problem: Problem[SolutionType, FitnessType], **kwargs):
        """
        Initializes the solver with a given problem and algorithm-specific
        parameters.

        Args:
            problem (Problem[SolutionType]): The optimization problem to solve.
            **kwargs: Algorithm-specific parameters.
        """
        self.problem = problem

    @abstractmethod
    def step(self) -> None:
        """
        Performs one iteration/generation of the optimization algorithm.
        """
        pass

    @abstractmethod
    def get_best(self) -> Tuple[SolutionType, FitnessType]:
        """
        Returns the best solution and its corresponding objective value(s) found
        so far.

        Returns:
            Tuple[SolutionType, List[float]]: A tuple containing:
                - The best solution (of type SolutionType).
                - The objective value(s) of the best solution.
        """
        pass
