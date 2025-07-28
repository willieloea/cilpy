# visualize_mpb.py
#!/usr/bin/env python3

"""
A script to run and visualize the Moving Peaks Benchmark (MPB).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D type hint

sys.path.append('.')
from cilpy.problem.mpb import MovingPeaksBenchmark

def run_visualization():
    """Initializes and runs the MPB visualization."""

    mpb = MovingPeaksBenchmark(
        dimension=2,
        num_peaks=8,
        min_coord=0.0,
        max_coord=100.0,
        min_height=30.0,
        max_height=70.0,
        min_width=5.0,
        max_width=15.0,
        change_frequency=1,
        change_severity=1.5,
        height_severity=5.0,
        width_severity=1.0,
        lambda_param=0.1,
    )

    fig = plt.figure(figsize=(10, 8))
    # Add a type hint for ax to help Pylance
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    
    grid_resolution = 100
    x_coords = np.linspace(mpb.get_bounds()[0][0], mpb.get_bounds()[1][0], grid_resolution)
    y_coords = np.linspace(mpb.get_bounds()[0][1], mpb.get_bounds()[1][1], grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Vectorized landscape calculation for performance
    def calculate_landscape_vectorized():
        # Create a (N*N, 2) array of all grid points
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        # Calculate all values at once
        z_values = np.array([mpb._get_raw_maximization_value(pos) for pos in positions])
        # Reshape back to grid form
        return z_values.reshape(X.shape)

    # 3. Define the animation update function
    def update(frame_num):
        # CORRECT LOGIC:
        # 1. Advance the evaluation counter.
        mpb._eval_count += 1
        # 2. Explicitly trigger the environment change based on the new count.
        mpb.change_environment()
        
        # 3. Calculate the Z values for the *new, static* landscape.
        Z = calculate_landscape_vectorized()

        # 4. Update the plot
        ax.clear()
        # Use modern cmap call and get the created artist
        surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
        
        ax.set_title(f"Moving Peaks Benchmark (Time Step: {frame_num})")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Fitness")
        ax.set_zlim(0, mpb.max_height + 20)
        
        print(f"Generated frame {frame_num}")
        
        # CORRECT: Return an iterable of artists that were drawn
        return (surface,)

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

    plt.show()

if __name__ == "__main__":
    run_visualization()