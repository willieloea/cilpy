import numpy as np

from .functions import Ackley

lower = np.array([-32, -32])
upper = np.array([32, 32])
my_func = Ackley(2, (lower, upper))

print(my_func.name)
print(my_func(np.array([2, 2])))


from .functions import Sphere

lower = np.array([-5.12, -5.12])
upper = np.array([5.12, 5.12])
my_func = Sphere(2, (lower, upper))

print(my_func.name)
print(my_func(np.array([2, 2])))