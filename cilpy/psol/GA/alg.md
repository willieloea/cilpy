Pseudo-code for the canonical GA:  
```pseudocode
n_s <- Randomly initialise n_x-dimensional candidate solutions
n_s <- Evaluate(n_s)
REPEAT
    parents <- Selection(n_s)           // select parents
    offspring <- Reproduction(parents)  // create offspring
    offspring <- Mutate(offspring)      // mutate offspring
    offspring <- Evaluate(offspring)    // evaluate offspring
    n_s <- Combine(n_s , offspring)     // next population
UNTIL stopping condition(s) satisfied
```