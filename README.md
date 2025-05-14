# `cilpy` (Compuational Intelligence Library for Python)
`cilpy` is a computational intelligence library written in Python. 
The library focuses on nature-inspired optimization algorithms and the tools related to those algorithms.

## Design principles:
 * Genericity
    * Problems: single-solution, multi-solution
    * Objectives: single objectives, multi- and many-objectives, static objective functions, dynamically changing search landscapes,
    * Constraints: boundary constrained and constrained problems,
    * Population: single population and multi-population algorithms, and 
    * hyper-heuristic algorithms
 * Extendability (abstract base classes)
 * Maintainability (good software engineering practices)

## Library outline
`./cilpy/core/` contains abstract base classes that can be used to interface with the frameworks of the library  
`./cilpy/fw/` contains the three primary frameworks of `cilpy`:
 1. `./cilpy/fw/solving/` a framework for solving (constrained) optimization problems,
 2. `./cilpy/fw/generation/` a framework for generating constrained optimization problems, and
 3. `./cilpy/fw/benchmark/` a framework for comparing various optimization algorithms.