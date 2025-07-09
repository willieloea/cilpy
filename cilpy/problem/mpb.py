# cilpy/problem/mpb.py

import random
from typing import Callable, List, Tuple

from . import Problem
from . import helpers

# =============================================================================
# Peak class to represent a single peak in the landscape
# =============================================================================

class _Peak:
    """Represents a single peak in the Moving Peaks Benchmark."""
    def __init__(self, position: List[float], height: float, width: float):
        self.v = position                   # Peak location vector
        self.h = height                     # Peak height
        self.w = width                      # Peak width
        self.s_v = [0.0] * len(position)    # Shift vector

    def evaluate(self, x: List[float]) -> float:
        """
        Calculates the peak's value at position x.

        p_i(x) = h - w * distance(x, v)
        """
        dist = helpers.distance(x, self.v)
        return self.h - self.w * dist

    def update(self,
               height_sev: float,
               width_sev: float,
               change_sev: float,
               lambda_param: float,
               bounds: Tuple[float, float],
               dim: int):
        """
        Updates the peak's environment parameters, height, width, and position.
        """
        # Update environment parameters
        # Generate a random vector p_r normalized to length 'change_sev'
        p_r = [random.uniform(-1, 1) for _ in range(dim)]
        mag_pr = helpers.magnitude(p_r)
        if mag_pr > 0:
            p_r = helpers.scale(p_r, change_sev / mag_pr)
        else:   # A zero vector
            p_r = [0.0] * dim

        # Calculate the new shift vector s_v
        correlated_move = helpers.scale(self.s_v, lambda_param)
        random_move = helpers.scale(p_r, 1.0 - lambda_param)
        combined_move = helpers.add(random_move, correlated_move)

        mag_combined = helpers.magnitude(combined_move)
        if mag_combined > 0:
            self.s_v = helpers.scale(combined_move, change_sev / mag_combined)
        else:
            self.s_v = [0.0] * dim

        # Update height
        self.h += height_sev * random.gauss(0, 1)

        # Update width
        self.w += width_sev * random.gauss(0, 1)

        # Update position
        self.v = helpers.add(self.v, self.s_v)

        # Enforce boundary conditions (reflecting boundaries)
        min_b, max_b = bounds
        for i in range(dim):
            if self.v[i] < min_b:
                self.v[i] = 2 * min_b - self.v[i]
                self.s_v[i] *= -1.0
            elif self.v[i] > max_b:
                self.v[i] = 2 * max_b - self.v[i]
                self.s_v[i] *= -1.0


# =============================================================================
# Main MovingPeaksBenchmark class implementing the Problem interface
# =============================================================================

class MovingPeaksBenchmark(Problem[List[float]]):
    """
    An implementation of the Moving Peaks Benchmark (MPB) generator.

    This class generates a dynamic optimization problem landscape consisting of
    several cone-shaped peaks that can change height, width, and position over
    time. It adheres to the `cilpy.problem.Problem` interface.

    Reference:
    Branke, J. (2001). "Evolutionary Optimization in Dynamic Environments".
    """

    def __init__(self,
                 num_peaks: int=10,
                 dimension: int=2,
                 min_coord: float=0.0,
                 max_coord: float=100.0,
                 min_height: float=30.0,
                 max_height: float=70.0,
                 min_width: float=1.0,
                 max_width: float=12.0,
                 change_frequency: int=500,
                 change_severity: float=1.0,
                 height_severity: float=7.0,
                 width_severity: float=1.0,
                 lambda_param: float=0.0,
                 problem_name: str="MovingPeaksBenchmark"):
        """
        Initializes the Moving Peaks Benchmark generator.

        Args:
            num_peaks: Number of peaks in the landscape.
            dimension: Dimensionality of the search space.
            min_coord: Lower bound for each dimension of the search space.
            max_coord: Upper bound for each dimension of the search space.
            min_height: Lower bound for the initial height of peaks.
            max_height: Upper bound for the initial height of peaks.
            min_width: Lower bound for the initial width of peaks.
            max_width: Upper bound for the initial width of peaks.
            change_frequency: Number of fitness evaluations between environment
                              changes.
            change_severity: The severity of peak movement.
            height_severity: The severity of peak height changes.
            width_severity: The severity of peak width changes.
            lambda_param: Correlation coefficient for peak movement. 0.0 for
                          random, 1.0 for linear movement.
            problem_name: A name for the problem instance.
        """
        self._dimension = dimension
        self._bounds = (min_coord, max_coord)
        self._name = problem_name
        self._change_frequency = change_frequency

        # Severity parameters for updates
        self._change_sev = change_severity
        self._height_sev = height_severity
        self._width_sev = width_severity
        self._lambda = lambda_param

        # Initialize peaks
        self.peaks: List[_Peak] = []
        for _ in range(num_peaks):
            pos = [random.uniform(min_coord, max_coord) for _ in range(dimension)]
            height = random.uniform(min_height, max_height)
            width = random.uniform(min_width, max_width)
            self.peaks.append(_Peak(pos, height, width))
        
        # Base landscape value B
        self.base_value = 0.0

    def _get_raw_maximization_value(self, x: List[float]) -> float:
        """
        Calculates the raw fitness of a solution x.
        This is the true maximization value of the landscape.
        F(x, t) = max{B, p_0(x, e_0), ..., p_n(x, e_n)}
        """
        peak_values = [p.evaluate(x) for p in self.peaks]
        return max([self.base_value] + peak_values)

    def _fitness(self, x: List[float]) -> float:
        """
        Calculates the fitness for a minimization solver by negating the
        raw maximization value.
        """
        return -self._get_raw_maximization_value(x)

    def get_objective_functions(self) -> List[Callable[[List[float]], float]]:
        return [self._fitness]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return ([], []) # MPB is unconstrained

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        min_b, max_b = self._bounds
        return ([min_b] * self._dimension, [max_b] * self._dimension)

    def get_dimension(self) -> int:
        return self._dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False) # Dynamic objective, static constraints

    def change_environment(self, iteration: int) -> None:
        """
        Updates the peak landscape if the change frequency is met.
        """
        if self._change_frequency > 0 and \
            iteration % self._change_frequency == 0:
            for peak in self.peaks:
                peak.update(
                    height_sev=self._height_sev,
                    width_sev=self._width_sev,
                    change_sev=self._change_sev,
                    lambda_param=self._lambda,
                    bounds=self._bounds,
                    dim=self._dimension
                )

    def initialize_solution(self) -> List[float]:
        min_b, max_b = self._bounds
        return [random.uniform(min_b, max_b) for _ in range(self._dimension)]

    @property
    def is_multiobjective(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name