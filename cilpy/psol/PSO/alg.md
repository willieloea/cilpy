Pseudo-code for the canonical PSO:
```pseudocode
Create and initialise an x_n-dimensional swarm
REPEAT
    FOR EACH particle i = 1, 2, ..., n_s DO
        IF f(x_i) is better than f(y_i) THEN
            Assign y_i = x_i
        END IF
        IF f(y_i) is better than f(y_h) THEN
            Assign y_h = y_i
        END IF
    END FOR
    FOR EACH particle i = 1, 2, ..., n_s DO
        update velocity
    END FOR
    FOR EACH particle 𝑖 = 1, 2, ..., n_s DO
        update position
    END FOR
UNTIL stopping condition is true
```