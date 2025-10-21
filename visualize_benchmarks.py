# visualize_benchmarks.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# Import your custom modules
from cilpy.problem.mpb import MovingPeaksBenchmark, generate_mpb_configs

# --- Core Visualization Functions ---

def visualize_mpb(problem: MovingPeaksBenchmark, resolution: int, frames: int, evals_per_frame: int):
    """
    Creates an animation of a 2D Moving Peaks Benchmark landscape.

    Args:
        problem (MovingPeaksBenchmark): An instance of the MPB.
        resolution (int): The number of points to sample along each axis.
        frames (int): The number of frames to render in the animation.
        evals_per_frame (int): The number of dummy evaluations to run between frames
            to advance the problem's internal state.
    """
    if problem.dimension != 2:
        raise ValueError("Visualization is only supported for 2D problems.")

    # 1. Setup the plot
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"Moving Peaks Benchmark: {problem.name}", fontsize=16)
    
    # 2. Create the grid for plotting
    min_b, max_b = problem.bounds
    x = np.linspace(min_b[0], max_b[0], resolution)
    y = np.linspace(min_b[1], max_b[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 3. Create the initial plot object (heatmap)
    # We use -Z because the problem negates the fitness for minimization
    im = ax.imshow(-Z, extent=(min_b[0], max_b[0], min_b[1], max_b[1]),
                   origin='lower',
                   cmap=mpl.colormaps['viridis'])
    fig.colorbar(im, ax=ax, label="Fitness Value")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    dynamic_title = ax.set_title("Evaluations: 0")

    # 4. Define the animation update function
    def update(frame):
        # Advance the environment state by performing dummy evaluations
        for _ in range(evals_per_frame):
            problem.evaluate(np.array([0.0, 0.0]))

        # Re-evaluate the entire grid to get the new landscape
        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                Z[i, j] = problem.evaluate(pos).fitness
        
        # Update the plot data and title
        im.set_data(-Z) # Use -Z for correct visualization
        total_evals = (frame + 1) * evals_per_frame
        dynamic_title.set_text(f"Evaluations: {total_evals}")
        
        # You may need to adjust color limits if peaks change height drastically
        # im.set_clim(vmin, vmax) 
        
        return [im, dynamic_title]

    # 5. Create and show the animation
    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
    plt.show()


# =============================================================================
# --- MAIN CONFIGURATION ---
# =============================================================================
if __name__ == "__main__":
    # --- CHOOSE WHICH BENCHMARK TO VISUALIZE ---
    BENCHMARK_TO_RUN = "MPB"
    
    # --- CHOOSE THE PROBLEM CLASS ---
    # Any of the 28 acronyms (e.g., "A1L", "P3R", "C2C", "STA", "C3R")
    PROBLEM_ACRONYM = "C3R" 
    
    # --- VISUALIZATION PARAMETERS ---
    GRID_RESOLUTION = 50  # Number of points per axis (e.g., 50x50 grid)
    ANIMATION_FRAMES = 3 # How many landscape changes to show
    EVALS_PER_FRAME = 500 # How many evaluations between each frame

    # --- Generate all problem configurations ---
    all_problem_configs = generate_mpb_configs(s_for_random=2.0)

    # Get the chosen configuration
    if PROBLEM_ACRONYM not in all_problem_configs:
        raise KeyError(f"Problem acronym '{PROBLEM_ACRONYM}' is not valid.")
    
    config = all_problem_configs[PROBLEM_ACRONYM]
    
    # Ensure config is 2D for visualization
    config["dimension"] = 2

    # --- RUN THE VISUALIZATION ---
    if BENCHMARK_TO_RUN == "MPB":
        mpb_problem = MovingPeaksBenchmark(**config)
        visualize_mpb(mpb_problem, GRID_RESOLUTION, ANIMATION_FRAMES, EVALS_PER_FRAME)

    else:
        print(f"Unknown benchmark: {BENCHMARK_TO_RUN}. Choose 'MPB' or 'CMPB'.")
