# cilpy/problem/plotter.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from typing import cast

from .functions import Sphere
from .functions import Ackley

# --- Plotting Functions ---

def plot_2d(func_class, resolution=500):
    """
    Plots a given function in 2 dimensions (d=1).

    Args:
        func_class: The class of the function to plot (e.g., Ackley, Sphere).
        resolution (int): The number of points to plot along the x-axis.
    """
    # Initialize the function for 1 dimension
    dimension = 1
    lower = np.full(dimension, -5.12)
    upper = np.full(dimension, 5.12)

    f = func_class(dimension=dimension, bounds=(lower, upper))

    # Get the input domain for the x-axis
    x_min = f.bounds[0][0]
    x_max = f.bounds[1][0]
    x = np.linspace(x_min, x_max, resolution)
    
    # Calculate y values by applying the function to each x
    # Note: The function expects an iterable, so we pass [i]
    y = np.array([f([i]) for i in x])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"2D Plot of {f.name} Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_3d(func_class, resolution=100):
    """
    Plots a given function as a surface in 3 dimensions (d=2).

    Args:
        func_class: The class of the function to plot (e.g., Ackley, Sphere).
        resolution (int): The number of points for the grid in each dimension.
    """
    # Initialize the function for 2 dimensions
    dimension = 2
    lower = np.full(dimension, -5.12)
    upper = np.full(dimension, 5.12)
    f = func_class(dimension=dimension, bounds=(lower, upper))
    
    # Create a grid of (x, y) points
    x_min, x_max = f.bounds[0][0], f.bounds[1][0]
    y_min, y_max = f.bounds[0][1], f.bounds[1][1]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values for each (x, y) point on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(point)
            
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    
    # Plot the surface with a colormap
    surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap("viridis"), antialiased=True)
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(f"3D Surface Plot of {f.name} Function")
    
    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


if __name__ == '__main__':
    print("Generating 2D plot for the Sphere function...")
    # plot_2d(Ackley)
    # plot_2d(Sphere)

    print("Generating 3D plot for the Sphere function...")
    # plot_3d(Ackley)
    plot_3d(Sphere)
