# cilpy/problem/mpb_gen.py

# This file implements the moving peaks benchmark generator to generate DOPs

from . import Problem
from typing import List, Optional, Tuple

import math
import random

class Peak:
    """
    Represents a single peak in the Moving Peaks Benchmark landscape.
    """
    def __init__(self,
                 height: float,
                 width: float,
                 position: List[float],
                 shift_vector: List[float]):
        self.height = height
        self.width = width
        self.position = position
        self.shift_vector = shift_vector

    def __repr__(self) -> str:
        pos_str = ", ".join(f"{p:.2f}" for p in self.position)
        return (f"Peak(h={self.height:.2f}, w={self.width:.2f}, "
                f"pos=[{pos_str}])")


class MovingPeaksBenchmark(Problem):
    """
    A function generator for the Moving Peaks Benchmark (MPB).
    """
    _spare_normal: Optional[float] = None

    def __init__(self,
                 num_peaks: int=10,
                 num_dims: int=5,
                 min_coord: float=0.0,
                 max_coord: float=100.0,
                 min_height: float=30.0,
                 max_height: float=70.0,
                 min_width: float=1.0,
                 max_width: float=12.0,
                 h_severity: float=7.0,
                 w_severity: float=1.0,
                 change_severity_s: float=1.0,
                 lambda_coeff: float=0.0,
                 base_function_b: float=0.0,
                 rand_seed: Optional[int]=None):
        """
        Initializes the MPB generator.

        Args:
            ... (all previous args)
            rand_seed (Optional[int]): Seed for the random number generator
                to ensure independent and reproducible streams of randomness.
        """
        self.num_peaks = num_peaks
        self._dimension = num_dims
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width
        self.h_severity = h_severity
        self.w_severity = w_severity
        self.change_severity_s = change_severity_s
        self.lambda_coeff = lambda_coeff
        self.base_function_b = base_function_b

        # Each MPB instance gets its own random number generator.
        self.rand = random.Random(rand_seed)

        self.peaks: List[Peak] = []
        for _ in range(self.num_peaks):
            height = self.rand.uniform(self.min_height, self.max_height)
            width = self.rand.uniform(self.min_width, self.max_width)
            position = [self.rand.uniform(self.min_coord, self.max_coord) for _ in range(self._dimension)]
            shift_vector = [0.0] * self._dimension
            self.peaks.append(Peak(height, width, position, shift_vector))

    @staticmethod
    def _vector_sub(v1: List[float], v2: List[float]) -> List[float]:
        return [a - b for a, b in zip(v1, v2)]

    @staticmethod
    def _vector_add(v1: List[float], v2: List[float]) -> List[float]:
        return [a + b for a, b in zip(v1, v2)]

    @staticmethod
    def _vector_mul_scalar(v: List[float], s: float) -> List[float]:
        return [a * s for a in v]

    @staticmethod
    def _vector_norm(v: List[float]) -> float:
        return math.sqrt(sum(c**2 for c in v))

    def _evaluate_single_peak(self, x: List[float], peak: Peak) -> float:
        diff = self._vector_sub(x, peak.position)
        squared_dist = sum(c**2 for c in diff)
        return peak.height - peak.width * math.sqrt(squared_dist)

    def evaluate(self, solution: List[float]) -> Tuple[List[float], List[float]]:
        if len(solution) != self._dimension:
            raise ValueError(f"Input vector x has dimension {len(solution)}, but "
                             f"problem is configured for dimension {self._dimension}.")
        peak_values = [self._evaluate_single_peak(solution, peak) for peak in self.peaks]
        peak_values.append(self.base_function_b)
        return [max(peak_values)], []

    def _generate_normal_random(self) -> float:
        if self._spare_normal is not None:
            result = self._spare_normal
            self._spare_normal = None
            return result
        u1 = self.rand.random()
        u2 = self.rand.random()
        mag = math.sqrt(-2.0 * math.log(u1))
        z1 = mag * math.cos(2.0 * math.pi * u2)
        z2 = mag * math.sin(2.0 * math.pi * u2)
        self._spare_normal = z2
        return z1

    def change_environment(self, iteration: int) -> None:
        for peak in self.peaks:
            sigma_t = self._generate_normal_random()
            peak.height += self.h_severity * sigma_t
            peak.width += self.w_severity * sigma_t
            peak.height = max(self.min_height, min(self.max_height, peak.height))
            peak.width = max(self.min_width, min(self.max_width, peak.width))

            p_r = [self._generate_normal_random() for _ in range(self._dimension)]
            norm_pr = self._vector_norm(p_r)
            if norm_pr > 0:
                p_r = self._vector_mul_scalar(p_r, self.change_severity_s / norm_pr)

            term1 = self._vector_mul_scalar(p_r, 1.0 - self.lambda_coeff)
            term2 = self._vector_mul_scalar(peak.shift_vector, self.lambda_coeff)
            combined_vector = self._vector_add(term1, term2)
            norm_combined = self._vector_norm(combined_vector)
            if norm_combined > 0:
                new_shift_vector = self._vector_mul_scalar(combined_vector, self.change_severity_s / norm_combined)
            else:
                new_shift_vector = [0.0] * self._dimension
            peak.shift_vector = new_shift_vector

            new_position = self._vector_add(peak.position, peak.shift_vector)
            for d in range(self._dimension):
                if new_position[d] < self.min_coord:
                    new_position[d] = self.min_coord + (self.min_coord - new_position[d])
                    peak.shift_vector[d] *= -1.0
                elif new_position[d] > self.max_coord:
                    new_position[d] = self.max_coord - (new_position[d] - self.max_coord)
                    peak.shift_vector[d] *= -1.0
            peak.position = new_position

