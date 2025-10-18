"""
The solver module.

This module defines the abstract interface for problem solvers within the
cilpy library, ensuring a consistent contract for all optimization algorithms.
"""
# cilpy/solver/__init__.py

from abc import ABC, abstractmethod
from typing import Generic, Tuple, List
from ..problem import Problem, Evaluation, SolutionType, FitnessType


class Solver(ABC, Generic[SolutionType, FitnessType]):
    """An abstract interface for a problem solver.

    All solvers in `cilpy.solver` should implement this interface. The interface
    is generic to support different solution representations (e.g., List[float],
    List[int], custom objects) and various fitness types (e.g., float for
    single-objective, List[float] for multi-objective).

    Attributes:
        problem (Problem[SolutionType, FitnessType]): The optimization problem
            instance that this solver is configured to solve.
        name (str): A string containing the name of the solver.
    """

    def __init__(self,
                 problem: Problem[SolutionType, FitnessType],
                 name: str,
                 **kwargs):
        """
        Initializes the solver with a given problem and algorithm-specific
        parameters.

        Args:
            problem (Problem[SolutionType, FitnessType]): The optimization
                problem to solve, which must implement the `Problem` interface.
            **kwargs: Algorithm-specific parameters that can be passed during
                solver initialization (e.g., `swarm_size`, `c1`, `c2` for PSO,
                `population_size`, `mutation_rate` for GA).
            name (str): The name of the solver.
        """
        self.problem = problem
        self.name = name

    @abstractmethod
    def step(self) -> None:
        """Performs one iteration or generation of the optimization algorithm.

        This method encapsulates the core logic for advancing the search process
        by one step, which might involve updating populations, particle
        positions, or other algorithm-specific internal states.
        """

        pass

    @abstractmethod
    def get_result(self) -> List[Tuple[SolutionType, Evaluation[FitnessType]]]:
        """Returns the final result(s) of the optimization process.

        This method provides the best solution(s) found by the solver along with
        their full evaluation, allowing for consistent retrieval of results
        regardless of the solver type or problem complexity.

        Returns:
            List[Tuple[SolutionType, Evaluation[FitnessType]]]: A list of tuples,
                where each tuple contains:
                - The solution (of type `SolutionType`).
                - The `Evaluation` object for that solution, including its
                  fitness value(s) and constraint violation information.
                For single-objective solvers, this list typically contains a
                single `(solution, evaluation)` tuple representing the best
                found.
                For multi-/many-objective solvers, this list represents the
                archive of non-dominated solutions (e.g., the Pareto front).
        """

        pass
