# cilpy/problem/mpb.py
"""
The Moving Peaks Benchmark (MPB) is a function generator that produces dynamic
optimization problems. Problem instances are produced by a function generator
that contains independent peaks within a multi-dimensional problem landscape.

A candidate solution (x) is evaluated at time t as follows:
  F(x, t) = max{B, p0(x, e0), p1(x, e2), ..., pn(x, en)}
where pi is an individual peak function, defined by the set of peak parameters,
ei. The value of B defines the basis function landscape, which has a default
value of 0 because MPB is a maximization problem.

=== _Peak ===
peak parameters, e_i, is a record like structure that records the following:
min_height
max_height
min_width
max_width
bounds/domain
peak_location
peak_height
peak_width
shift_vector


=== MovingPeaksBenchmark ===
h_severity: determine scaling factors for peak height adjustment
w_severity: determine scaling factors for peak width adjustment
change_severity: constant determining amount of peak change between landscapes
sigma(t) ~ N(0, 1): random variable for stochastic updates
lambda: coefficient to scale the amount of random peak movement.
          - large lambda -> little random movement
          - small lambda -> a lot of random movement
"""

import numpy as np
import random
from typing import Callable, List, Tuple

from . import Problem

class _Peak:
    """Represents a single peak in the Moving Peaks Benchmark."""

    def __init__(self, position: np.ndarray, height: float, width: float):
        """
        Constructor for a peak in the Moving Peaks Benchmark.

        Params:
            s_v (int | float): The shift vector influences the movement of the peak during an environment change.

        Args:
            position (int | float): defines the center of a peak and is used to set the value for `self.v`
            height (int | float): defines the height of a peak and is used to set the value for `self.h`
            width (int | float): defines the width of the peak and is used to set the value for `self.w`
        """
        self.v = position  # Peak location vector
        self.h = height    # Peak height
        self.w = width     # Peak width
        self.s_v = np.zeros_like(position)  # Shift vector

    def evaluate(self, x: np.ndarray) -> np.float64:
        """
        Calculates the peak's value at position x.
        p_i(x) = h - w * ||x - v||
               = h - w * [Sum_i abs(a_i)]^0.5, where a_i = x_i - v_i
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
        Updates the peak's environment parameters (height, width, position).
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

    The MPB generator produces maximizing dynamic optimization problems. The MPB
    generates several independent peaks that change height, width, and position,
    based on the input parameters. A candidate solution is evaluated as the
    maximum between all peaks and some base value.
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
        name: str = "MovingPeaksBenchmark",
    ):
        """
        Constructor for the Moving Peaks Benchmark.

        Args:
            dimension (int): the dimensions of the search landscape
            num_peaks (int): the number of peaks in the search landscape
            min_coord (float): the lower bound of all dimensions in the search landscape
            max_coord (float): the upper bound of all dimensions in the search landscape
            min_height (float): the minimum height a peak may have
            max_height (float): the maximum height a peak may have
            min_width (float): the minimum width a peak may have
            max_width (float): the maximum width a peak may have
            change_frequency (float): the frequency at which the search landscape changes
            change_severity (float): controls how severely peak positions change
            height_severity (float): controls how severely peak heights change
            width_severity (float): controls how severely peak widths change
            lambda_param (float): 
        """
        min_bounds = np.array([min_coord] * dimension)
        max_bounds = np.array([max_coord] * dimension)

        super().__init__(dimension, (min_bounds, max_bounds), name)

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

    def _fitness(self, x: np.ndarray) -> np.float64:
        """
        Calculates the fitness for a maximization solver.
        """
        self._eval_count += 1
        if self._change_frequency > 0 and self._eval_count > 0 and \
           self._eval_count % self._change_frequency == 0:
            for peak in self.peaks:
                peak.update(
                    height_sev=self._height_sev,
                    width_sev=self._width_sev,
                    change_sev=self._change_sev,
                    lambda_param=self._lambda,
                    bounds=self.bounds,
                    dim=self.dimension,
                )
        peak_values = [p.evaluate(x) for p in self.peaks]
        return np.float64(max([self.base_value] + peak_values))

    def get_objective_functions(self) -> List[Callable[[np.ndarray], np.float64]]:
        return [self._fitness]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return ([], [])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)
