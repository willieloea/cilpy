# calculate_pred.py
import argparse
import glob
import pandas as pd
import numpy as np
import sys

def calculate_pred_for_file(filepath: str, change_frequency: int) -> pd.Series:
    """
    Calculates the P_RED metric for each run in a single experiment data file.

    Args:
        filepath (str): The path to the CSV output file from ExperimentRunner.
        change_frequency (int): The number of iterations between environmental changes.

    Returns:
        pd.Series: A pandas Series containing the P_RED value for each run.
    """
    try:
        df = pd.read_csv(filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Skipping empty or missing file: {filepath}")
        return pd.Series(dtype=float)

    # 1. Validate that the required columns for P_RED exist and are complete.
    required_cols = ['optimum_value', 'worst_value']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: File '{filepath}' is missing required columns for P_RED calculation.")
        return pd.Series(dtype=float)

    # Coerce to numeric, turning our empty strings ('') into NaN
    df['optimum_value'] = pd.to_numeric(df['optimum_value'], errors='coerce')
    df['worst_value'] = pd.to_numeric(df['worst_value'], errors='coerce')

    # Check if there are any missing values (NaNs) after coercion
    if df['optimum_value'].isnull().any() or df['worst_value'].isnull().any():
        print(
            f"\nError: Cannot calculate P_RED for '{filepath}'.\n"
            "The problem used in this experiment does not provide known optimum and worst values.\n"
            "To fix this, implement the `get_optimum_value()` and `get_worst_value()` methods\n"
            "in your Problem class.\n",
            file=sys.stderr
        )
        # We exit here because this is a fundamental failure for this script's purpose
        sys.exit(1)

    # --- 2. Calculate Relative Error (P_RE) for each data point ---
    # P_RE = (f_best(t) - f_min(t)) / (f_max(t) - f_min(t))
    f_best = df['best_fitness']
    f_min = df['optimum_value']
    f_max = df['worst_value']

    denominator = f_max - f_min

    # Avoid division by zero. If max and min are the same, error is 0 if the
    # solver found the optimum, otherwise it's undefined (treat as max error = 1).
    p_re = np.divide(
        f_best - f_min,
        denominator,
        out=np.full_like(denominator, 1.0), # Default to 1.0 (max error)
        where=(denominator != 0)
    )

    # Clamp P_RE values to be between 0 and 1, as f_best could exceed f_max
    df['p_re'] = np.clip(p_re, 0, 1)

    # --- 2. Sample P_RE at each change period to create the performance vector ---
    # We sample just before the change occurs, at iterations `change_frequency`,
    # `2*change_frequency`, etc.
    sample_iterations = np.arange(change_frequency, df['iteration'].max() + 1, change_frequency)
    df_sampled = df[df['iteration'].isin(sample_iterations)]

    # --- 3. Calculate P_RED for each run ---
    # P_RED(b) = sqrt( sum((1 - b_i)^2) / n_v )
    # where b is the vector of P_RE values for a run.
    pred_values = []
    # Group by run to process each independent trial
    for run_id, run_data in df_sampled.groupby('run'):
        b = run_data['p_re'].to_numpy()  # This is the performance vector

        
        if len(b) == 0:
            print(f"Warning: No samples found for run {run_id} in {filepath}. "
                  "Check if max_iterations > change_frequency.")
            continue
            
        squared_errors = (1 - b)**2
        mean_squared_error = np.mean(squared_errors)
        p_red = np.sqrt(mean_squared_error)
        pred_values.append(p_red)

    return pd.Series(pred_values)

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate the P_RED metric from CILpy experiment output files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='+',
        help="A list of file paths or a glob pattern (e.g., '*.out.csv') "
             "to the experiment output files."
    )
    parser.add_argument(
        '--change_freq',
        type=int,
        required=True,
        help="The change frequency (number of iterations) used in the experiment."
    )
    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print("Error: No files found matching the provided patterns.")
        return

    print(f"Analyzing {len(all_files)} files with change frequency = {args.change_freq}\n")

    results = []
    for f in sorted(all_files):
        print(f"Processing: {f}")
        pred_series = calculate_pred_for_file(f, args.change_freq)
        
        if not pred_series.empty:
            results.append({
                'File': f,
                'Mean P_RED': pred_series.mean(),
                'Std Dev P_RED': pred_series.std(),
                'Num Runs': len(pred_series)
            })

    if not results:
        print("\nNo results could be calculated.")
        return

    # --- Display Results ---
    results_df = pd.DataFrame(results)
    print("\n--- P_RED Metric Results ---")
    print("Lower Mean P_RED indicates better performance (closer to the ideal profile).")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
