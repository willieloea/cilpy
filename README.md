# `cilpy` (Compuational Intelligence Library for Python)
The Compuational Intelligence Library for Python (`cilpy`) is a computational
intelligence library written in Python to ease the process of conducting
experiments on nature-inspired algorithms (NIAs) for constrained optimization
problems and multi-objective optimization problems.

The `cilpy` library consists of three components:
 1. A problem generation component (`cilpy.problem`)
 2. A problem solving component (`cilpy.solver`)
 3. A solution comparison component (`cilpy.compare`)

If a user implements the interface defined for each of these components,
`cilpy` can run an experiment, saving effort in studying NIAs. For a guide on
how `cilpy` works and how it can be used, read `./docs/quickstart.md`

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

## Requirements
### Core Requirements
* [X] The library `MUST` support multi-objective and constrained optimization
problems.
* [X] The library `MUST` support different computational intelligence paradigms
(such as genetic algorithms and particle swarm optimization).
* [X] The library `MUST` be easily extendable to make integration with other
code easy.
* [ ] The library `MUST` have good documetation.
* [ ] The library `MUST` be easy to maintain - the following `MAY` be
provided/followed in this vein:
    * [X] Version control `MUST` be used.
    * [X] A consistent code style `MUST` be defined and followed.
    * [ ] Documentation for development `MUST` be provided.
    * [ ] Unit tests and integration tests `SHOULD` be provided.
* [X] The library `MUST` be developed in Python.
* [ ] The library `MUST` be very generic to allow for:
    * [X] single-solution problems
    * [ ] multi-solution problems
    * [X] single objectives
    * [X] multi- and many-objectives
    * [X] static objective functions
    * [X] dynamically changing search landscapes
    * [X] boundary constrained problems
    * [X] constrained problems
    * [X] single population algorithms
    * [ ] multi-population algorithms
    * [ ] hyper-heuristic algorithms
* [ ] The library `SHOULD` consider interaction with future liraries, including
ones used for:
    * [ ] empirical analysis
    * [ ] fitness landscape
    * [ ] results repositories
* [ ] The library `MUST` allow for various constraint handling techniques
including:
    * [ ] techniques ensuring feasibility of solutions throughout the search
    process.
    * [ ] techniques allowing infeasible solutions during the search process,
    while applying repair mechanisms later.
    * [X] techniques which formulate the constrained optimization problem as a
    box-constrained optimization problem through the use of penalty methods.
    * [X] techniques which formulate the constrained optimization problem as a
    dual Lagrangian.
    * [ ] techniques which formulate the constrained optimization problem as a
    box-constrained multi-/many-objective optimization problem, and then to use
    multi-/many-objective optimization problem to find feasible solutions.
* [X] The library `MUST` allow constraint handling techniques which are
algorithm agnostic to be applied to any meta-heruistic.
* [ ] The library `SHOULD` satisfy fellow students and supervisors.
* [X] The library `MAY` contribute to an existing library, rather than being
written from scratch.

### Ancillary Requirements
These requirements provide support to the primary goals of the library.
* [ ] A number of constraint-handling approaches `MUST` be implemented as proof
of concept.

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