import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple

# To run this script, ensure that the 'cilpy' directory is in your Python path.
# This can be achieved by running this script from the same directory that
# contains the 'cilpy' folder.
from cilpy.problem.mpb import MovingPeaksBenchmark, generate_mpb_configs

def get_current_landscape(
    problem: MovingPeaksBenchmark,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> np.ndarray:
    """
    Calculates the fitness for each point on a 2D grid without advancing
    the problem's internal evaluation counter.

    This provides a 'snapshot' of the current landscape state for visualization.

    Args:
        problem (MovingPeaksBenchmark): The MPB instance to visualize.
        x_coords (np.ndarray): A meshgrid of x-coordinates.
        y_coords (np.ndarray): A meshgrid of y-coordinates.

    Returns:
        np.ndarray: A 2D array of fitness values corresponding to the grid.
    """
    if problem.dimension != 2:
        raise ValueError("Visualization is only supported for 2D problems.")

    # Initialize a grid to store fitness values (Z-axis)
    fitness_grid = np.zeros_like(x_coords)

    # Calculate the fitness for each (x, y) point on the grid
    for i in range(x_coords.shape[0]):
        for j in range(x_coords.shape[1]):
            solution = np.array([x_coords[i, j], y_coords[i, j]])

            # This logic mimics the core evaluation of the MPB function
            # without calling the problem's 'evaluate' method, thus preventing
            # the environment from changing while we render a single frame.
            peak_values = [p.evaluate(solution) for p in problem.peaks]
            max_value = float(max([0.0] + peak_values))
            
            # Negate the value to match the minimization objective
            fitness_grid[i, j] = -max_value

    return fitness_grid

def main():
    """
    Sets up and runs the MPB visualization.
    """
    # 1. --- Problem Configuration ---
    all_configs = generate_mpb_configs(dimension=2, min_width=8)
    config = all_configs['C2R']
    config['change_frequency'] = 100 # more frequent change

    # Instantiate the problem
    mpb_problem = MovingPeaksBenchmark(**config)
    
    # 2. --- Grid and Visualization Setup ---
    # Define the resolution of the plot
    grid_points = 150
    domain_min, domain_max = mpb_problem.bounds[0][0], mpb_problem.bounds[1][0]

    # Create a grid of points to evaluate
    x = np.linspace(domain_min, domain_max, grid_points)
    y = np.linspace(domain_min, domain_max, grid_points)
    X, Y = np.meshgrid(x, y)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 7))
    # fig.suptitle("Moving Peaks Benchmark Visualization", fontsize=16)

    # Get the initial landscape
    Z = get_current_landscape(mpb_problem, X, Y)

    # Create the plot. 'imshow' displays the data as an image.
    # The 'magma' colormap shows low values as dark colors, as requested.
    im = ax.imshow(
        Z,
        cmap='magma',
        origin='lower',
        extent=(domain_min, domain_max, domain_min, domain_max),
        vmin=-mpb_problem.peaks[0].h, # Set a reasonable initial color limit
        vmax=0
    )
    
    # Add a color bar to show the fitness scale
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fitness Value")
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    title = ax.set_title(f"Environment: 0 | Evaluations: 0")

    # 3. --- Animation Logic ---
    def update(frame: int) -> Tuple:
        """
        This function is called for each frame of the animation. It advances
        the MPB environment and redraws the landscape.
        """
        # Advance the environment by 'change_frequency' evaluations.
        # We use a dummy solution as the input to the evaluate function,
        # since we only care about incrementing the internal counter.
        dummy_solution = mpb_problem.bounds[0]
        evals_per_frame = mpb_problem._change_frequency

        if evals_per_frame > 0:
            for _ in range(evals_per_frame):
                mpb_problem.evaluate(dummy_solution)

        # Get the new landscape snapshot
        Z_new = get_current_landscape(mpb_problem, X, Y)
        
        # Update the plot data and color limits
        im.set_data(Z_new)
        im.set_clim(np.min(Z_new), 0) # Adjust color limits to the new data range

        # Update the title
        env_num = mpb_problem._eval_count // evals_per_frame if evals_per_frame > 0 else 0
        title.set_text(f"Environment: {env_num} | Evaluations: {mpb_problem._eval_count}")
        
        print(f"Rendered frame {frame} (Environment #{env_num})")

        return (im, title)

    # Create and run the animation
    # 'frames' determines how many times the environment will change.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=10,
        interval=1000,  # Milliseconds between frames
        blit=False,
        repeat=False
    )
    
    plt.show()

if __name__ == '__main__':
    main()
