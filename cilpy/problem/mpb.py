# cilpy/problem/mpb.py

import numpy as np
import random
from typing import Callable, List, Tuple

from . import Problem

class _Peak:
    """Represents a single peak in the Moving Peaks Benchmark using numpy."""

    def __init__(self, position: np.ndarray, height: float, width: float):
        self.v = position  # Peak location vector (numpy array)
        self.h = height    # Peak height
        self.w = width     # Peak width
        self.s_v = np.zeros_like(position)  # Shift vector (numpy array)

    def evaluate(self, x: np.ndarray) -> np.float64:
        """
        Calculates the peak's value at position x.
        p_i(x) = h - w * ||x - v||
        """
        dist = np.linalg.norm(x - self.v)
        return np.float64(self.h - self.w * dist)

    def update(
        self,
        height_sev: float,
        width_sev: float,
        change_sev: float,
        lambda_param: float,
        bounds: Tuple[np.ndarray, np.ndarray],
        dim: int,
    ):
        """
        Updates the peak's environment parameters (height, width, position)
        using numpy vectorized operations.
        """
        # Generate a random vector p_r and normalize to length 'change_sev'
        p_r = np.random.uniform(-1, 1, size=dim)
        mag_pr = np.linalg.norm(p_r)
        if mag_pr > 0:
            p_r *= change_sev / mag_pr

        # Calculate the new shift vector s_v based on Equation 4.4
        combined_move = (1.0 - lambda_param) * p_r + lambda_param * self.s_v
        mag_combined = np.linalg.norm(combined_move)
        if mag_combined > 0:
            self.s_v = combined_move * (change_sev / mag_combined)
        else:
            self.s_v = np.zeros(dim)

        # Update height and width with Gaussian noise
        self.h += height_sev * np.random.normal(0, 1)
        self.w += width_sev * np.random.normal(0, 1)

        # Update position
        self.v += self.s_v

        # Enforce boundary conditions using numpy's boolean indexing
        min_b, max_b = bounds
        low_mask = self.v < min_b
        self.v[low_mask] = 2 * min_b[low_mask] - self.v[low_mask]
        self.s_v[low_mask] *= -1.0
        high_mask = self.v > max_b
        self.v[high_mask] = 2 * max_b[high_mask] - self.v[high_mask]
        self.s_v[high_mask] *= -1.0


# =============================================================================
# Main MovingPeaksBenchmark class implementing the Problem interface
# =============================================================================


class MovingPeaksBenchmark(Problem[np.ndarray, np.float64]):
    """
    An implementation of the Moving Peaks Benchmark (MPB) generator using numpy.
    """

    def __init__(
        self,
        dimension: int = 2,
        num_peaks: int = 10,
        min_coord: float = 0.0,
        max_coord: float = 100.0,
        min_height: float = 30.0,
        max_height: float = 70.0,
        min_width: float = 1.0,
        max_width: float = 12.0,
        change_frequency: int = 5000,
        change_severity: float = 1.0,
        height_severity: float = 7.0,
        width_severity: float = 1.0,
        lambda_param: float = 0.0,
        problem_name: str = "MovingPeaksBenchmark",
    ):
        min_bounds = np.array([min_coord] * dimension)
        max_bounds = np.array([max_coord] * dimension)
        # Assuming your ABC accepts these kwargs
        super().__init__(dimension=dimension, bounds=(min_bounds, max_bounds))

        self._dimension = dimension
        self._bounds = (min_bounds, max_bounds)
        self._name = problem_name
        self._change_frequency = change_frequency
        
        self.max_height = max_height

        self._change_sev = change_severity
        self._height_sev = height_severity
        self._width_sev = width_severity
        self._lambda = lambda_param

        self.peaks: List[_Peak] = []
        for _ in range(num_peaks):
            pos = np.random.uniform(min_coord, max_coord, size=dimension)
            height = random.uniform(min_height, max_height)
            width = random.uniform(min_width, max_width)
            self.peaks.append(_Peak(pos, height, width))

        self.base_value = 0.0
        self._eval_count = 0

    def _get_raw_maximization_value(self, x: np.ndarray) -> np.float64:
        """
        Calculates the raw fitness of a solution x.
        """
        peak_values = [p.evaluate(x) for p in self.peaks]
        return np.float64(max([self.base_value] + peak_values))

    def _fitness(self, x: np.ndarray) -> np.float64:
        """
        Calculates the fitness for a minimization solver.
        """
        self._eval_count += 1
        self.change_environment()
        return -self._get_raw_maximization_value(x)

    def get_objective_functions(self) -> List[Callable[[np.ndarray], np.float64]]:
        return [self._fitness]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return ([], [])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def get_dimension(self) -> int:
        return self._dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)

    def change_environment(self) -> None:
        """
        Updates the peak landscape if the change frequency is met.
        """
        if self._change_frequency > 0 and self._eval_count > 0 and \
           self._eval_count % self._change_frequency == 0:
            for peak in self.peaks:
                peak.update(
                    height_sev=self._height_sev,
                    width_sev=self._width_sev,
                    change_sev=self._change_sev,
                    lambda_param=self._lambda,
                    bounds=self._bounds,
                    dim=self._dimension,
                )

    def initialize_solution(self) -> np.ndarray:
        min_b, max_b = self._bounds
        return np.random.uniform(min_b, max_b, size=self._dimension)

    @property
    def is_multiobjective(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name