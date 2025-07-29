# visualize_mpb.py
#!/usr/bin/env python3

"""
A script to run and visualize the Moving Peaks Benchmark (MPB).
This version is updated to work with the new Problem interface where
environment changes are triggered by fitness evaluations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# This block allows running the script from the project's root directory
# without having to install the `cilpy` package.
sys.path.append('.')
from cilpy.problem.mpb import MovingPeaksBenchmark

def run_visualization():
    """Initializes and runs the MPB visualization."""

    # 1. Initialize the MPB problem.
    #    CRITICAL CHANGE: Set `change_frequency=1` to make the environment
    #    change on every fitness evaluation, which we'll trigger once per frame.
    mpb = MovingPeaksBenchmark(
        dimension=2,
        num_peaks=8,
        min_coord=0.0,
        max_coord=100.0,
        min_height=30.0,
        max_height=70.0,
        min_width=5.0,
        max_width=15.0,
        change_frequency=1,  # <--- THIS IS THE KEY
        change_severity=1.5,
        height_severity=5.0,
        width_severity=1.0,
        lambda_param=0.1,
    )

    # Get the objective function from the problem's public interface
    objective_func = mpb.get_objective_functions()[0]

    fig = plt.figure(figsize=(10, 8))
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    
    # Setup the grid for plotting
    grid_resolution = 100
    x_coords = np.linspace(mpb.get_bounds()[0][0], mpb.get_bounds()[1][0], grid_resolution)
    y_coords = np.linspace(mpb.get_bounds()[0][1], mpb.get_bounds()[1][1], grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Vectorized function to calculate landscape values for plotting.
    # This uses the internal `_get_raw_maximization_value` which does NOT trigger a change.
    def calculate_landscape_vectorized():
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        z_values = np.array([mpb._get_raw_maximization_value(pos) for pos in positions])
        return z_values.reshape(X.shape)

    # 2. Define the animation update function
    def update(frame_num):
        objective_func(np.zeros(mpb.get_dimension()))
        
        # B. Calculate the Z values for the *newly updated* landscape for plotting.
        Z = calculate_landscape_vectorized()

        # C. Update the plot with the new landscape
        ax.clear()
        surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
        
        ax.set_title(f"Moving Peaks Benchmark (Time Step: {frame_num})")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Fitness")
        ax.set_zlim(0, mpb.max_height + 20)
        
        print(f"Generated frame {frame_num}")
        
        return (surface,)

    # Create and run the animation
    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    run_visualization()