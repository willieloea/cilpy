# Requirements
## Core Requirements
* [ ] The library `MUST` support multi-objective and constrained optimization
problems.
* [ ] The library `MUST` support different computational intelligence paradigms
(such as genetic algorithms and particle swarm optimization).
* [ ] The library `MUST` be easily extendable to not make integration with other
code difficult.
* [ ] The library `MUST` have good documetation.
* [ ] The library `MUST` have good testing.
* [ ] The library `MUST` be easy to maintain - the following `MAY` be
provided/followed in this vein:
    * [ ] Version control `MUST` be used.
    * [ ] A consistent code style `MUST` be defined and followed.
    * [ ] Documentation for development `MUST` be provided.
    * [ ] Unit tests and integration tests `SHOULD` be provided.
* [ ] The library `MUST` be developed in Python.
* [ ] The library `MUST` be very generic to allow for:
    * [ ] single-solution problems
    * [ ] multi-solution problems
    * [ ] single objectives
    * [ ] multi- and many-objectives
    * [ ] static objective functions
    * [ ] dynamically changing search landscapes
    * [ ] boundary constrained problems
    * [ ] constrained problems
    * [ ] single population algorithms
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
    * [ ] techniques which formulate the constrained optimization problem as a
    box-constrained optimization problem through the use of penalty methods.
    * [ ] techniques which formulate the constrained optimization problem as a
    dual Lagrangian.
    * [ ] techniques which formulate the constrained optimization problem as a
    box-constrained multi-/many-objective optimization problem, and then to use
    multi-/many-objective optimization problem to find feasible solutions.
* [ ] The library `MUST` allow constraint handling techniques which are
algorithm agnostic to be applied to any meta-heruistic.
* [ ] The library `SHOULD` satisfy fellow students and supervisors.
* [ ] The library `MAY` contribute to an existing library, rather than being
written from scratch.

## Ancillary Requirements
These requirements provide support to the primary goals of the library.
* [ ] A number of constraint-handling approaches `MUST` be implemented as proof
of concept.