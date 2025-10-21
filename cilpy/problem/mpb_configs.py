"""
Generates parameter configurations for the Moving Peaks Benchmark (MPB).

This module programmatically creates 28 distinct problem classes based on the
comprehensive classification scheme proposed by Duhain and Engelbrecht. This
scheme combines three different ways of categorizing dynamic optimization
problems (DOPs), resulting in 27 dynamic classes and 1 static class.

The final classes are identified by a 3-letter acronym (e.g., 'A1L'), which is
formed by combining one code from each of the three classification schemes
below.

---

### 1. Duhain & Engelbrecht: Spatial and Temporal Severity (First Letter)

This classification defines the magnitude and frequency of changes in the landscape.

-   **Progressive ('P')**: Frequent but small changes. Optima move gradually.
    -   `change_frequency`: Low (e.g., 1000) -> High temporal change.
    -   `change_severity` (`s`): Low value.
    -   `height_severity`: Low value.

-   **Abrupt ('A')**: Infrequent but large changes. The landscape is stable for
    long periods, then changes drastically.
    -   `change_frequency`: High (e.g., 5000) -> Low temporal change.
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

-   **Chaotic ('C')**: Frequent and large changes. The landscape changes
    drastically and often.
    -   `change_frequency`: Low (e.g., 1000) -> High temporal change.
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

### 2. Hu & Eberhart / Shi & Eberhart: Optima Modification (Second Letter)

This classification defines *what* changes about the peaks: their position,
their height (value), or both.

-   **Type I ('1')**: Peak locations change, but their heights remain constant.
    -   `height_severity`: 0.0
    -   Requires `change_severity` (`s`) != 0 to allow movement.

-   **Type II ('2')**: Peak locations are static, but their heights change.
    -   `height_severity`: Non-zero value (determined by Severity class).
    -   Requires `change_severity` (`s`) = 0 to prevent movement.

-   **Type III ('3')**: Both peak locations and heights change.
    -   `height_severity`: Non-zero value.
    -   Requires `change_severity` (`s`) != 0.

### 3. Angeline: Optima Trajectory (Third Letter)

This classification defines the *pattern* of the peaks' movement over time.

-   **Linear ('L')**: Peaks move in a straight, correlated line.
    -   `lambda_param`: 1.0
    -   Requires `change_severity` (`s`) != 0 for movement to occur.

-   **Circular ('C')**: Peaks move in a circular or periodic pattern. This is
    achieved by preventing translational movement.
    -   Requires `change_severity` (`s`) = 0.
    -   (Note: A full implementation would apply a rotation matrix, but for
      parameterization, `s=0` is the key.)

-   **Random ('R')**: Peaks move randomly without a discernible pattern.
    -   `lambda_param`: 0.0
    -   Requires `change_severity` (`s`) != 0 for movement to occur.

### Combination and Conflict Resolution

A 3-letter acronym combines one choice from each category (e.g., `A1L` is
Abrupt, Type I, Linear).

A conflict arises when combining rules with contradictory requirements for the
`change_severity` (`s`) parameter. For example:
-   Movement: **Linear ('L')** requires `s != 0`.
-   Modification: **Type II ('2')** requires `s = 0`.
-   Resulting class **`*2L`** is impossible.

This script handles these exceptions by catching them, and overriding the
parameter values.

### The Static Class ('STA')

A 28th class, 'STA', is included for a completely static environment. All
parameters related to change (`change_frequency`, `change_severity`,
`height_severity`, etc.) are set to 0.
"""

# import itertools
# from typing import Any, Dict, Tuple

# # --- Example Usage ---
# if __name__ == "__main__":