# How do I implement the following problems in python:
# 1. static constrained static optimization problems (SCSO),
# 2. static constrained dynamic optimization problems (SCDO),
# 3. dynamic constrained static optimization problems (DCSO),
# 4. dynamic constrained dynamic optimization problems (DCDO).

# === SCSO === #
# This file demonstrates how to implement constrained optimization problems in
# Python.

# The standard form of a constrained minimization problem is:
# 
# minimize f(x)
# subject to:
#   g_i(x) <= 0, for i =       1, ..., m_g       (inequality constraints)
#   h_j(x) = 0,  for j = m_g + 1, ..., m_g + m_h (equality constraints)
# where `x` is a vector of decision variables

def check_dims(x: list[int], expected_dims: int):
    if len(x) != expected_dims:
        raise ValueError(f"Input vector x must have {expected_dims} dimensions")


class UnconstrainedProblem:
    """Represents an unconstrained optimization problem."""
    def __init__(self, dimensions):
        self.dimensions = dimensions
    
    def objective_function(self, x) -> int:
        """Sphere Function"""
        check_dims(x, self.dimensions)
        return sum(val**2 for val in x)

class EqualityConstrainedProblem:
    """
    Represents a constrained optimization problem with an equality constraint.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def objective_function(self, x) -> int:
        """Sphere Function"""
        check_dims(x, self.dimensions)
        return sum(val**2 for val in x)
    
    def equality_constraints(self, x) -> list[int]:
        """Constraint: sum(x_i) - 1 = 0"""
        check_dims(x, self.dimensions)
        h1 = sum(x) - 1
        return [h1]

class InequalityConstrainedProblem:
    """
    Represents a constrained optimization problem with an inequality constraint.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def objective_function(self, x) -> int:
        """Sphere Function"""
        check_dims(x, self.dimensions)
        return sum(val**2 for val in x)

    def inequality_constraints(self, x) -> list[int]:
        """Constraint: 2 - sum(x_i) <= 0"""
        check_dims(x, self.dimensions)
        g1 = 2 - sum(x)
        return [g1]

class ConstrainedProblem:
    """
    Represents a constrained optimization problem with both an equality and
    inequality constraint.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def objective_function(self, x) -> int:
        """Sphere Function"""
        check_dims(x, self.dimensions)
        return sum(val**2 for val in x)
    
    def equality_constraints(self, x) -> list[int]:
        """Constraint: sum(x_i) - 1 = 0"""
        check_dims(x, self.dimensions)
        h1 = sum(x) - 1
        return [h1]

    def inequality_constraints(self, x) -> list[int]:
        """
        Constraints: 
         # 2 - sum(x_i) <= 0
         # sum(x_i^2) - 4 <= 0
        """
        check_dims(x, self.dimensions)
        g1 = 2 - sum(x)
        g2 = sum(val**2 for val in x) - 4
        return [g1, g2]

if __name__ == '__main__':
    dim = 2
    sol0 = [0.0, 0.0]
    sol1 = [0.5, 0.5]
    sol2 = [1.0, 1.0]
    sol3 = [0.0, 1.5]

    solutions = [sol0, sol1, sol2, sol3]

    for i, solution in enumerate(solutions):
        print(f"--- For problem {i} ---")
        p1 = UnconstrainedProblem(dim)
        obj_val = p1.objective_function(solution)
        print(f"Objective value at {solution}: {obj_val:.4f}")

        p2 = EqualityConstrainedProblem(dim)
        obj_val = p2.objective_function(solution)
        # print(f"Objective value at {solution}: {obj_val:.4f}")
        print(f"Equality violations: {p2.equality_constraints(solution)}")

        p3 = InequalityConstrainedProblem(dim)
        obj_val = p3.objective_function(solution)
        # print(f"Objective value at {solution}: {obj_val:.4f}")
        print(f"Inequality violations: {p3.inequality_constraints(solution)}")

        p4 = ConstrainedProblem(dim)
        obj_val = p4.objective_function(solution)
        # print(f"Objective value at {solution}: {obj_val:.4f}")
        # print(f"Equality violations: {p4.equality_constraints(solution)}")
        print(f"Inequality violations: {p4.inequality_constraints(solution)}\n")

# === SCDO === #
# === DCSO === #
# === DCDO === #