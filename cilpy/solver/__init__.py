from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, List
from ..problem import Problem

# Define a generic type for solutions
SolutionType = TypeVar('SolutionType')

class Solver(ABC, Generic[SolutionType]):
    """
    An abstract interface for a problem solver.

    All solvers in `cilpy.solver` should implement this interface. The interface
    is generic to support different solution representations (e.g., List[float],
    List[int], custom objects).
    """
    def __init__(self, problem: Problem[SolutionType], **kwargs):
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
    def get_best(self) -> Tuple[SolutionType, List[float]]:
        """
        Returns the best solution and its corresponding objective value(s) found
        so far.

        Returns:
            Tuple[SolutionType, List[float]]: A tuple containing:
                - The best solution (of type SolutionType).
                - The objective value(s) of the best solution.
        """
        pass

