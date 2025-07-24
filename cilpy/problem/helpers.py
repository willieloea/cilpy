# cilpy/problem/helpers.py

# This file contains helper functions for vector mathematics, to avoid needing
# NumPy as a dependency

import math
from typing import List


def dot(v1: List[float], v2: List[float]) -> float:
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))


def magnitude(v: List[float]) -> float:
    """Calculates the magnitude (Euclidean norm) of a vector."""
    return math.sqrt(dot(v, v))


def subtract(v1: List[float], v2: List[float]) -> List[float]:
    """Subtracts vector v2 from v1."""
    return [x - y for x, y in zip(v1, v2)]


def add(v1: List[float], v2: List[float]) -> List[float]:
    """Adds two vectors."""
    return [x + y for x, y in zip(v1, v2)]


def scale(v: List[float], s: float) -> List[float]:
    """Scales a vector by a scalar value."""
    return [x * s for x in v]


def distance(v1: List[float], v2: List[float]) -> float:
    """Calculates the Euclidean distance between two vectors."""
    return magnitude(subtract(v1, v2))
