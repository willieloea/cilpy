# analyze_final_results.py
import argparse
import glob
import pandas as pd
import numpy as np

def analyze_file(filepath: str) -> dict:
    """
    Analyzes a single experiment file to find the average minimum and feasibility count.

    Args:
        filepath (str): The path to the CSV output file.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    try:
        df = pd.read_csv(filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {} # Return empty dict if file is missing or empty

    # Find the last row for each independent run
    # This gives us the final result of each of the 30 runs
    final_results_df = df.loc[df.groupby('run')['iteration'].idxmax()]

    # Total number of completed runs
    total_runs = len(final_results_df)
    if total_runs == 0:
        return {}

    # Q1: Over all runs, what was the average minimum (final best fitness)?
    avg_minimum = final_results_df['best_fitness'].mean()
    std_minimum = final_results_df['best_fitness'].std()

    # Q2: Over all runs, how many minimums were feasible?
    # We sum the 'is_feasible' column (1 for feasible, 0 for not)
    feasible_count = final_results_df['is_feasible'].sum()

    return {
        'File': filepath,
        'Total Runs': total_runs,
        'Avg. Final Fitness': avg_minimum,
        'Std. Dev. Fitness': std_minimum,
        'Feasible Count': int(feasible_count)
    }

def main():
    """Main function to parse arguments and run the final results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze the final results from CILpy experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='+',
        help="A list of file paths or a glob pattern (e.g., '*.out.csv') "
             "to the experiment output files."
    )
    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print("Error: No files found matching the provided patterns.")
        return

    print(f"Analyzing final results from {len(all_files)} files...\n")

    results = []
    for f in sorted(all_files):
        analysis = analyze_file(f)
        if analysis:
            results.append(analysis)

    if not results:
        print("No valid data could be analyzed.")
        return

    # --- Display Results in a formatted table ---
    results_df = pd.DataFrame(results)
    print("--- Final Results Summary ---")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
