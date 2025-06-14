# `cilpy.pgen`
The `pgen` component of `cilpy` is used to generate optimization problems which
can be used to generate optimization problems for the `psol` and `cmpr`
components of `cilpy`.

# Optimization problems
## Constrained optimization problems
There are many ways in which constrained optimization problems (COPs) can be
categorized, but for now `cilpy.pgen` categorizes COPs by whether constraints
are static or dynamic, and whether the objective function is static or dynamic.
Thus the following problem categories exist:
 * static constrained static optimization problems (SCSO),
 * static constrained dynamic optimization problems (SCDO),
 * dynamic constrained static optimization problems (DCSO), and
 * dynamic constrained dynamic optimization problems (DCDO).
