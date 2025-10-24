import pandas as pd
import numpy as np

def calculate_pred(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    
    # Store PRE vectors for each run
    pre_vectors = []

    for run_id in df['run'].unique():
        run_df = df[df['run'] == run_id].copy()
        
        f_best = run_df['best_fitness']
        f_min = run_df['global_optimum_fitness']
        f_max = run_df['global_anti_optimum_fitness']

        # Calculate PRE for the run, handling division by zero
        denominator = f_max - f_min
        # Use np.divide for safe division
        pre_vector = np.divide(
            f_best - f_min,
            denominator,
            out=np.zeros_like(denominator, dtype=float), # Default to 0 if denominator is 0
            where=(denominator != 0)
        )
        pre_vectors.append(pre_vector)

    # Now calculate PRED for each vector
    pred_values = []
    for b in pre_vectors:
        nv = len(b)
        if nv == 0:
            pred_values.append(np.nan)
            continue
        
        # PRED(b) = sqrt( sum( (1 - b_i)^2 ) / n_v )
        sum_of_squares = np.sum((1 - b)**2)
        pred = np.sqrt(sum_of_squares / nv)
        pred_values.append(pred)

    # You can then report the mean and std dev of PRED across all runs
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)

    print(f"File: {csv_file_path}")
    print(f"Mean Relative Error Distance (PRED): {mean_pred:.4f}")
    print(f"Std Dev of PRED: {std_pred:.4f}")
    
# Example usage:
calculate_pred('CMPB_A3L_A3L_RIGA_AlphaConstraint.out.csv')
calculate_pred('CMPB_A3L_A3L_RIGA_CCLS.out.csv')
