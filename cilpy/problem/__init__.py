from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Tuple, TypeVar

# Generic type for solutions
SolutionType = TypeVar('SolutionType')

class Problem(ABC, Generic[SolutionType]):
    """
    An abstract interface for optimization problems in cilpy.

    All problems in `cilpy.problem` should implement this interface to ensure
    compatibility with solvers and comparison tools. The interface is generic
    to support different solution representations (e.g., List[float], List[int],
    custom objects).
    """
    
    @abstractmethod
    def get_objective_functions(self) -> List[Callable[[SolutionType], float]]:
        """
        Returns the objective function(s) of a problem.
        """
        pass

    @abstractmethod
    def get_constraints(self) -> Tuple[List[Callable], List[Callable]]:
        """
        Returns the constraint functions of a problem.

        Returns:
            Tuple[List[Callable], List[Callable]]: A tuple containing:
                - List of inequality constraints
                - List of equality constraints
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Any:
        """
        Returns the search space boundaries or constraints for the problem.

        Returns:
            Any: The bounds or constraints defining the solution space (e.g.,
            Tuple[List[float], List[float]] for real-valued problems, or other
            structures for different solution types).
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Returns the dimensionality or size of the solution space.

        Returns:
            int: The number of dimensions or elements in a solution
                 (interpretation depends on solution type).
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> Tuple[bool, bool]:
        """
        Indicates whether the problem has dynamic objectives or constraints.

        Returns:
            Tuple[bool, bool]: A tuple of (is_objective_dynamic,
                               is_constraint_dynamic).
        """
        pass

    @abstractmethod
    def change_environment(self, iteration: int) -> None:
        """
        Updates the problem state for dynamic problems (e.g., change constraints
        or objectives).

        Args:
            iteration (int): The current iteration/generation of the solver.
        """
        pass

    @abstractmethod
    def initialize_solution(self) -> SolutionType:
        """
        Generates or defines the structure of an initial solution for the
        problem.

        Returns:
            SolutionType: An initial solution (e.g., a random List[float],
                          List[int], or custom object).
        """
        pass

    @property
    @abstractmethod
    def is_multiobjective(self) -> bool:
        """
        Indicates whether the problem is a multi-objective optimization problem.

        Returns:
            bool: True if the problem is a MOOP, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name or identifier of the problem.

        Returns:
            str: A string identifier for the problem.
        """
        pass
