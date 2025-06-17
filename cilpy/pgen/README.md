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

Eventually, the library should consider problems along all of the following
categories:
 * number of solutions        (single vs multi)
 * number of objectives       (single vs multi)
 * type of objective function (static vs dynamic)
 * number of populations      (single vs multi)
 * number of constraints      (none vs some)
 * type of constraints        (static vs dynamic)
 * type of algorithm          (NIA vs hyper-heuristic)
This would mean that there may be up to $2^7 = 128$ probelm categories.