Pseudo-code for the canonical DE:
```pseudocode
Randomly initialise a n_x-dimensional set of n_s candidate solutions
Evaluate the fitness for all candidate solutions
REPEAT
    FOR EACH x_i(t) in candidate solutions DO
        u_i(t) <- CreateTrialVector()
        x_i_p(t) <- Crossover(u_i(t), x_i(t))
        x_i(t+1) <- ReplaceFitness(x_i_p(t), x_i(t))
    END FOR
UNTIL stopping condition(s) satisfied
```