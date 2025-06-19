# `cilpy.psol`
The `psol` component of `cilpy` is used to solve optimization problems generated
by `cilpy.pgen`, and then produce output so that `cilpy.cmpr` can compare
different solutions.

## Requirements
 * This component SHOULD allow for solving the following classess of COPs:
    * static  constrained static  optimization problems (SCSO),
    * static  constrained dynamic optimization problems (SCDO),
    * dynamic constrained static  optimization problems (DCSO), and
    * dynamic constrained dynamic optimization problems (DCDO).  
 * This component MUST allow for the use of static and dynamic NIAs.
 * The project MUST implement the following NIAs as a proof of concept:
    * (SCSO) a standard inertia weight gbest PSO,
    * (SCDO) a quantum-inspired PSO,
    * (DCSO) a DE/rand/1/bin, and
    * (DCDO) any of the dynamic DEs. (One in Gary Pamparà's PhD thesis)
