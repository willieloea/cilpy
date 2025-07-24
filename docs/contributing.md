This file covers style, structure, documentation, and the process for adding new
components to `cilpy`.

# Contributing to `cilpy`

First off, thank you for considering contributing to `cilpy`! This project is
designed to be a collaborative and extensible tool for the computational
intelligence community. Your help is greatly appreciated.

This document provides guidelines for contributing to the project. Following
them helps maintain the quality and consistency of the codebase, making it
easier for everyone to read, use, and contribute to.

## Table of Contents
 * [How to Contribute](#how-to-contribute)
   * [Reporting Bugs](#reporting-bugs)
   * [Suggesting Enhancements](#suggesting-enhancements)
  <!-- * [How to Make a Code Contribution](#how-to-make-a-code-contribution) -->
 * [Coding Style and Conventions](#coding-style-and-conventions)
   * [General Style (PEP 8)](#general-style-pep-8)
   * [Naming Conventions](#naming-conventions)
   * [Type Hinting](#type-hinting)
   * [Docstrings](#docstrings)
   * [Imports](#imports)
   * [Core Dependencies](#core-dependencies)
 * [Adding New Components](#adding-new-components)
   * [Adding a New Solver](#adding-a-new-solver)
   * [Adding a New Problem](#adding-a-new-problem)
 * [Testing](#testing)
 * [Contribution Process](#contribution-process)

## How to Contribute

### Reporting Bugs
If you find any bugs, contact the current maintainer, Willie Loftie-Eaton,
through his email: [willieloea@gmail.com](mailto:willieloea@gmail.com)

### Suggesting Enhancements
If you have an idea for a new feature or an improvement to an existing one,
please contact the current maintainer, Willie Loftie-Eaton, through his email:
[willieloea@gmail.com](mailto:willieloea@gmail.com)

<!-- ### How to Make a Code Contribution -->

## Coding Style and Conventions
Consistency is key. All Python code contributed to `cilpy` must follow these
conventions.

### General Style (PEP 8)
 * All code should adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/),
 the official style guide for Python code.
 * We recommend using a linter like `flake8` or an autoformatter like `black` to
 automatically enforce these standards.
 * Use a maximum line length of 79 characters for code.
 * Use a maximum line length of 72 characters for docstrings and comments.

### Naming Conventions
 * **Modules**: `snake_case` (e.g., `file_name.py`).
 * **Classes**: `PascalCase` (e.g., `ClassName`).
 * **Functions & Methods**: `snake_case` (e.g., `function_name`).
 * **Variables**: `snake_case` (e.g., `variable_name`).
 * **Constants**: `ALL_CAPS` (e.g., `CONSTANT_CONST`).
 * **Internal Members**: Use a single leading underscore `_` for internal
 functions or methods that are not part of the public API (e.g.,
 `_internal_function`).

### Type Hinting
 * **Mandatory Type Hinting**: All function and method signatures **must**
 include type hints from the `typing` module. This is crucial for
 maintainability and static analysis.
 * Use the generic types defined in the interfaces where applicable (e.g.,
 `Problem[SolutionType]`).
 * See the `Problem` and `Solver` interfaces in `cilpy/problem/__init__.py` and
 `cilpy/solver/__init__.py` for canonical examples.

### Docstrings
 * All public modules, classes, functions, and methods must have a docstring.
 * We use the **Google-style docstring format**. It is readable and can be
 easily parsed by documentation generators.
 * A good docstring explains *what* the code does, its parameters, and what it
 returns.

**Example Google-style Docstring:**
```python
def __init__(self,
             problem: Problem[List[float]],
             swarm_size: int = 30,
             alpha_start: float = 1.0,
             alpha_end: float = 0.5,
             max_iterations: int = 1000):
    """Initializes the QPSO solver.

    Args:
        problem: The optimization problem to solve.
        swarm_size: Number of particles in the swarm.
        alpha_start: Initial value for the contraction-expansion coefficient.
        alpha_end: Final value for the contraction-expansion coefficient.
        max_iterations: The total number of iterations for the run. This is
                        required to schedule the linear decrease of alpha.
    """
    # ... implementation ...
```

### Imports
 * Imports should be grouped in the following order:
    1.  Standard library imports (e.g., `random`, `math`, `typing`).
    2.  Third-party library imports (if any).
    3.  Local application/library imports (e.g.,
    `from ..problem import Problem`).
 * Within the `cilpy` library, use relative imports to refer to other parts of
 the library (e.g., `from ..problem import Problem` inside a solver file).

### Core Dependencies
For now, the library should have no dependencies. Contact Willie Loftie-Eaton
through email ([willieloea@gmail.com](mailto:willieloea@gmail.com)) if you think
this should change.

## Adding New Components
The library is designed for easy extension.

### Adding a New Solver
1.  Create a new file in `cilpy/solver/` with a `snake_case` name (e.g.,
`my_new_solver.py`).
2.  Import the base `Solver` class: `from . import Solver`.
3.  Create your solver class, inheriting from `Solver`. Specify the
`SolutionType` you are working with (e.g.,
`class MyNewSolver(Solver[List[float]]):`).
4.  Implement the required abstract methods: `__init__`, `step`, and `get_best`.
5.  Your `__init__` method must call `super().__init__(problem, **kwargs)`.
6.  Follow all coding style and documentation guidelines mentioned above.
7.  Add your new solver to `cilpy/solver/__init__.py` to make it easily
importable.

### Adding a New Problem
1.  Create a new file in `cilpy/problem/` (e.g., `my_new_problem.py`).
2.  Import the base `Problem` class: `from . import Problem`.
3.  Create your problem class, inheriting from `Problem`.
4.  Implement all required abstract methods and properties from the `Problem`
interface.
5.  Add your new problem to `cilpy/problem/__init__.py`.

## Testing
*   The `test/` directory contains tests for the library.
*   When you add a new feature or fix a bug, you must add corresponding tests.
*   Tests should be self-contained and not require manual intervention.
*   We aim to use the `pytest` framework for testing.

## Contribution Process
1.  Ensure your code lints without errors and follows the style guide.
2.  Make sure you have added or updated tests for your changes.
3.  Update the relevant documentation (docstrings, READMEs) if you have changed
any public APIs.
4.  Your contribution should have a clear, descriptive title (e.g., "Feature:
Add GVPSO Solver", "Fix: Off-by-one error in runner").
5.  As part of you contribution, provide a description explaining the "what" and
"why" of your changes.

Thank you again for your contribution