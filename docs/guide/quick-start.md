# Quick Start
The easiest way to see `cilpy` in action is to run one of the provided examples. This example runs a Particle Swarm Optimization (PSO) solver on the simple static Sphere function.
Navigate to the examples directory and run the script:
```bash
cd examples
python sphere_pso.py
```
You should see output like this:
```
--- Starting Experiment: GbestPSO on Sphere ---
Saving results to: output.csv
  Iteration 50/500 complete. Current Best Fitness: 0.1234
  Iteration 100/500 complete. Current Best Fitness: 0.0456
  ...
Results successfully saved.
```
This will generate an *.out.csv file with the results of the run.
