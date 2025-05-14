from abc import ABC, abstractmethod
import numpy as np

class OptimizationProblem(ABC):
    """
    Abstract Base Class for specifying a problem

    Attributes:
    -----------
    dim : int
        the number of dimensions of the problem

    num_objectives: int
        the number of objectives of the problem

    Methods:
    --------
    evaluate():
        evaluate the solution given
    """
    def __init__(self, dim: int, num_objectives: int):
        # set problem attributes like dimension, number of objectives, etc.
        pass

    @abstractmethod
    def evaluate(self, decision_variables: np.ndarray) -> tuple[np.ndarray]:
        """
        Evaluates the solution.
        Args:
            decision_variables: A numpy array of decision variables.
        Returns:
            objective_values: A numpy array of objective function values.
        """
        pass

    # other methods for constrained problems and dynamic problems