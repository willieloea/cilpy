# `cilpy` (Compuational Intelligence Library for Python)
`cilpy` is a computational intelligence library written in Python to ease the process of conducting experiments on nature-inspired algorithms (NIAs) for constrained optimization problems and multi-objective optimization problems.

**Repo Outline**:  
`cilpy/` contains the library  
`dev/` contains resources related to development and maintenance of the library  
`docs/` contains documentation for the library  
`examples/` contains examples on how to use the library  
`scripts/` contains scripts to help keep the library clean  
`test/` contains tests for the library  

For detailed information about what each of these directories contain, read the README in each directory.

## Design Principles:
 * **Genericity** - the library should enable experimentation with the following
    * Problems: single-solution, multi-solution
    * Objectives: single objectives, multi- and many-objectives, static
    objective functions, dynamically changing search landscapes
    * Constraints: boundary constrained and constrained problems,
    * Population: single population and multi-population algorithms, and 
    * hyper-heuristic algorithms
 * **Extendability** - the library should allow for users to easily extend the
 library to suit the needs of their experiments
 * **Maintainability** - the library (good software engineering practices)
