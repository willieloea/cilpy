The `cilpy` library is broken into three components:
 - problem
 - solver
 - comparison

Python files must follow Google-Style Docstrings.

Prefixes used for denoting branch purposes are as follows:
* `ft` - feature - feature development
* `fx/` – fix - bug fixes
* `ch/` – chore - maintenance tasks
* `rf/` - refactor – internal code improvements
* `hf/` – hotfix - urgent production fixes
* `ts/` – test - testing related branches

## Design Principles
The library has the following design principles:
 * **Genericity** - the library should enable experimentation with the following
    * Problems: single-solution, multi-solution
    * Objectives: single objectives, multi- and many-objectives, static
    objective functions, dynamically changing search landscapes
    * Constraints: boundary constrained and constrained problems,
    * Population: single population and multi-population algorithms, and 
    * hyper-heuristic algorithms.
 * **Extendability** - the library should allow for users to easily extend the
 library to suit the needs of their experiments.
 * **Maintainability** - the library be easy to maintain and contribute to, i.e.
  follow good software engineering practices.

## Contributing to `cilpy`
This section provides guidelines for contributing to the project. Following
them helps maintain the quality and consistency of the codebase, making it
easier for everyone to read, use, and contribute to.

### Coding Style and Conventions
#### General Style (PEP 8)
 * All code should adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/),
 the official style guide for Python code.
 * An autoformatter like `black` is recommended.
 * Use a maximum line length of 79 characters for code.
 * Use a maximum line length of 72 characters for docstrings and comments.
 * All public modules, classes, functions, and methods must have a docstring.


#### Naming Conventions
 * **Modules**: `snake_case` (e.g., `file_name.py`).
 * **Classes**: `PascalCase` (e.g., `ClassName`).
 * **Functions & Methods**: `snake_case` (e.g., `function_name`).
 * **Variables**: `snake_case` (e.g., `variable_name`).
 * **Constants**: `ALL_CAPS` (e.g., `CONSTANT_CONST`).
 * **Internal Members**: Use a single leading underscore `_` for internal
 functions or methods that are not part of the public API (e.g.,
 `_internal_function`).

#### Type Hinting
 * **Mandatory Type Hinting**: All function and method signatures **must**
 include type hints from the `typing` module. This is crucial for
 maintainability and debugging.
 * Use the generic types defined in the interfaces where applicable (e.g.,
 `Problem[SolutionType]`).
 * See the `Problem` and `Solver` interfaces in `cilpy/problem/__init__.py` and
 `cilpy/solver/__init__.py` for canonical examples.

#### Imports
 * Imports should be grouped in the following order:
    1.  Standard library imports (e.g., `random`, `math`, `typing`).
    2.  Third-party library imports (if any).
    3.  Local application/library imports (e.g.,
    `from ..problem import Problem`).
 * Within the `cilpy` library, use relative imports to refer to other parts of
 the library (e.g., `from ..problem import Problem` inside a solver file).

#### Core Dependencies
For now, the library should have no dependencies. Contact Willie Loftie-Eaton
through email ([willieloea@gmail.com](mailto:willieloea@gmail.com)) if you think
this should change.

### Adding New Components
TODO: Provide instructions for adding new problems, solvers, and comparisons to
the library.
