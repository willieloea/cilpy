# analyze_results.py
"""
py analyze_results.py CMPB_A*.out.csv
py analyze_results.py CMPB_P*.out.csv
py analyze_results.py CMPB_C*.out.csv

--- Final Results Summary ---
                                         File  Total Runs  Avg. Final Fitness  Std. Dev. Fitness  Feasible Count
CMPB_A1R_A1R_HyperMGA_AlphaConstraint.out.csv          30         -865.818022         545.250754              28
           CMPB_A1R_A1R_HyperMGA_CCLS.out.csv          30         -320.246584         258.000065              29
    CMPB_A1R_A1R_RIGA_AlphaConstraint.out.csv          30         -891.036783         200.543227              30
               CMPB_A1R_A1R_RIGA_CCLS.out.csv          30         -263.831082         210.661268              30
CMPB_A3R_A3R_HyperMGA_AlphaConstraint.out.csv          30        -1143.698781        1484.025773              15
           CMPB_A3R_A3R_HyperMGA_CCLS.out.csv          30        -2561.142788        1060.515536              30
    CMPB_A3R_A3R_RIGA_AlphaConstraint.out.csv          30        -1771.419200        1054.305963              28
               CMPB_A3R_A3R_RIGA_CCLS.out.csv          30         -174.247355         172.568552              23

--- Final Results Summary ---
                                         File  Total Runs  Avg. Final Fitness  Std. Dev. Fitness  Feasible Count
CMPB_P1R_P1R_HyperMGA_AlphaConstraint.out.csv          30        -4233.171774         644.796885              30
           CMPB_P1R_P1R_HyperMGA_CCLS.out.csv          30        -2397.280356        1243.981345              30
    CMPB_P1R_P1R_RIGA_AlphaConstraint.out.csv          30        -2279.124407         829.048109              30
               CMPB_P1R_P1R_RIGA_CCLS.out.csv          30         -243.958403         312.565119              25
CMPB_P3R_P3R_HyperMGA_AlphaConstraint.out.csv          30        -4687.436853        1034.802856              30
           CMPB_P3R_P3R_HyperMGA_CCLS.out.csv          30        -3583.599339        1223.842566              30
    CMPB_P3R_P3R_RIGA_AlphaConstraint.out.csv          30        -3619.954350         984.004327              30
               CMPB_P3R_P3R_RIGA_CCLS.out.csv          30         -583.626090         303.646449              30
Analyzing final results from 8 files...

--- Final Results Summary ---
                                         File  Total Runs  Avg. Final Fitness  Std. Dev. Fitness  Feasible Count
CMPB_C1R_C1R_HyperMGA_AlphaConstraint.out.csv          30        -2189.669159         942.544303              30
           CMPB_C1R_C1R_HyperMGA_CCLS.out.csv          30        -1815.150193         969.421461              30
    CMPB_C1R_C1R_RIGA_AlphaConstraint.out.csv          30        -2108.834877         731.605889              30
               CMPB_C1R_C1R_RIGA_CCLS.out.csv          30         -375.621147         377.246596              25
CMPB_C3R_C3R_HyperMGA_AlphaConstraint.out.csv          30        -1026.473774        1747.466126              10
           CMPB_C3R_C3R_HyperMGA_CCLS.out.csv          30        -1997.537766        1448.430357              23
    CMPB_C3R_C3R_RIGA_AlphaConstraint.out.csv          30        -3336.700532         863.013528              30
               CMPB_C3R_C3R_RIGA_CCLS.out.csv          30         -916.316072         875.161955              24
"""
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
