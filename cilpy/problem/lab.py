import numpy as np

from .functions import Sphere

# func = Sphere.get_objective_functions
lower = np.array([-5.12, -5.12])
upper = np.array([5.12, 5.12])
my_func = Sphere(2, (lower, upper))

print(my_func.name)
print(my_func([2,2]))