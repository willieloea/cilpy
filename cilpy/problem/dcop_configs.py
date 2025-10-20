import itertools
from pprint import pprint

def generate_mpb_configs(s_for_random: float = 1.0):
    """
    Programmatically generates parameter dictionaries for all 28 MPB classes.

    This function combines the rules from three classification schemes to generate
    27 dynamic problem configurations and 1 static configuration. It handles
    contradictions between rules as specified.

    Args:
        s_for_random (float): The non-zero value to use for the change_severity
            parameter `s` when a non-zero value is required. Defaults to 1.0.

    Returns:
        Dict[str, Dict]: A dictionary where keys are the 3-letter acronyms
            (e.g., "A1C", "P3L") and values are the corresponding parameter
            dictionaries for the MovingPeaksBenchmark constructor. A "STA" key
            is included for the static case.
    """
    if s_for_random == 0:
        raise ValueError("'s_for_random' must be a non-zero value.")

    # 1. Base Configuration (common to all classes)
    base_params = {
        "dimension": 5,
        "num_peaks": 10,
        "domain": (0.0, 100.0),
        "min_height": 30.0,
        "max_height": 70.0,
        "min_width": 1.0,
        "max_width": 12.0,
        "width_severity": 0.05, # Often kept low
    }

    # 2. Define "Low" vs. "High" Values for severity and frequency
    LOW_S, HIGH_S = s_for_random, 10.0
    LOW_H, HIGH_H = 7.0, 15.0
    
    # High temporal frequency = low number of evaluations between changes
    FREQ_PROGRESSIVE = 1000 
    FREQ_ABRUPT = 5000
    FREQ_CHAOTIC = 1000

    # 3. Classification Rules
    
    # Duhain & Engelbrecht: Severity (Spatial & Temporal)
    # Acronyms: P (Progressive), A (Abrupt), C (Chaotic)
    SEVERITY_CLASSES = {
        'P': {
            "change_severity": LOW_S, "height_severity": LOW_H, 
            "change_frequency": FREQ_PROGRESSIVE,
        },
        'A': {
            "change_severity": HIGH_S, "height_severity": HIGH_H,
            "change_frequency": FREQ_ABRUPT,
        },
        'C': {
            "change_severity": HIGH_S, "height_severity": HIGH_H,
            "change_frequency": FREQ_CHAOTIC,
        },
    }

    # Hu & Eberhart / Shi & Eberhart: Optima Modification
    # Acronyms: 1 (Type I), 2 (Type II), 3 (Type III)
    # We use 's_req' to define the requirement for the change_severity (s)
    MODIFICATION_CLASSES = {
        '1': {"height_severity": 0.0, "s_req": "!=0"},
        '2': {"s_req": "=0"}, # height_severity will be taken from SEVERITY_CLASSES
        '3': {"s_req": "!=0"}, # height_severity will be taken from SEVERITY_CLASSES
    }

    # Angeline: Optima Trajectory
    # Acronyms: L (Linear), C (Circular), R (Random)
    MOVEMENT_CLASSES = {
        'L': {"lambda_param": 1.0, "s_req": "!=0"},
        'C': {"lambda_param": 0.0, "s_req": "=0"}, # lambda is irrelevant when s=0
        'R': {"lambda_param": 0.0, "s_req": "!=0"},
    }

    # --- Generation Logic ---
    all_configs = {}
    
    # 4. Iterate through all 3*3*3 = 27 combinations
    severity_codes = SEVERITY_CLASSES.keys()
    modification_codes = MODIFICATION_CLASSES.keys()
    movement_codes = MOVEMENT_CLASSES.keys()
    
    for sev_code, mod_code, mov_code in itertools.product(severity_codes, modification_codes, movement_codes):
        acronym = f"{sev_code}{mod_code}{mov_code}"
        
        # Start with base and add severity parameters
        config = base_params.copy()
        config.update(SEVERITY_CLASSES[sev_code])
        config["name"] = acronym
        
        mod_rules = MODIFICATION_CLASSES[mod_code]
        mov_rules = MOVEMENT_CLASSES[mov_code]
        
        # 5. Resolve Conflicts for `change_severity` (s)
        s_req_mod = mod_rules['s_req']
        s_req_mov = mov_rules['s_req']
        
        is_conflict = (s_req_mod == "!=0" and s_req_mov == "=0") or \
                      (s_req_mod == "=0" and s_req_mov == "!=0")
        
        if is_conflict:
            config['change_severity'] = 'XXX' # Assign placeholder on conflict
        elif s_req_mod == "=0" or s_req_mov == "=0":
            # If either requires s=0 and there's no conflict, it must be 0
            config['change_severity'] = 0.0
        else:
            # Otherwise, s must be non-zero. Use the value from the severity class.
            # This is already set, but we make it explicit for clarity.
            pass

        # 6. Apply overrides from modification and movement rules
        if 'height_severity' in mod_rules:
            config['height_severity'] = mod_rules['height_severity']
            
        if 'lambda_param' in mov_rules:
            config['lambda_param'] = mov_rules['lambda_param']
            
        all_configs[acronym] = config

    # Add the static problem class
    static_config = base_params.copy()
    static_config.update({
        "change_frequency": 0,
        "change_severity": 0,
        "height_severity": 0,
        "width_severity": 0,
        "lambda_param": 0,
        "name": "STA"
    })
    all_configs["STA"] = static_config
    
    return all_configs

# --- Example Usage ---
if __name__ == "__main__":
    # You can specify what s != 0 should be. For example, 1.0
    mpb_problem_classes = generate_mpb_configs(s_for_random=1.0)
    
    print(f"Generated a total of {len(mpb_problem_classes)} problem classes.")
    
    print("\n--- Example: A non-conflicting class (A1L) ---")
    # Abrupt, Type I, Linear
    # - Abrupt sets s=10, h=15.
    # - Type I overrides h to 0. It requires s!=0.
    # - Linear sets lambda=1.0. It requires s!=0.
    # - No conflict on s. s remains 10.0 from Abrupt.
    pprint(mpb_problem_classes['A1L'])

    print("\n--- Example: A conflicting class (A2L) ---")
    # Abrupt, Type II, Linear
    # - Abrupt sets s=10, h=15.
    # - Type II requires s=0.
    # - Linear requires s!=0.
    # - CONFLICT! `change_severity` should be 'XXX'.
    pprint(mpb_problem_classes['A2L'])

    print("\n--- Example: A zero-s class (P2C) ---")
    # Progressive, Type II, Circular
    # - Progressive sets s=1.0, h=7.0.
    # - Type II requires s=0.
    # - Circular requires s=0.
    # - No conflict. `change_severity` becomes 0.0.
    pprint(mpb_problem_classes['P2C'])

    print("\n--- Example: The static class (STA) ---")
    pprint(mpb_problem_classes['STA'])

#     static_params = {
#         "dimension": 5,
#         "num_peaks": 10,
#         "change_frequency": 10,
#         "lambda_param": 0,
#     }

#     progressive_params = {
#         "dimension": 5,
#         "num_peaks": 10,
#         "change_frequency": 10,
#         "lambda_param": 0,
#     }

#     abrupt_params = {
#         "dimension": 5,
#         "num_peaks": 10,
#         "change_frequency": 10,
#         "lambda_param": 0,
#     }

#     chaotic_params = {
#         "dimension": 5,
#         "num_peaks": 10,
#         "change_frequency": 10,
#         "lambda_param": 0,
#     }

# """

# change_severity = [0, inf]
# height_severity = [0, inf]
# width_severity  = [0, 1]
# lambda          = [0, 1]

# MPB and classes:
# 1. Duhain & Engelbrecht: spatial and temporal changes
#     - __A__: at least one of $s, hSeverity$ and $wSeverity$ is set to a high value
#         "height_severity": 10,
#         "change_frequency" = 100,
#         "width_severity" = 0.05,
#     - __C__: $s, hSeverity$ and $wSeverity$ are high, and change is frequent
#         "change_severity": 10,
#         "height_severity": 10,
#         "width_severity" = 10,
#         "change_frequency" = 30,
#     - __P__: $s, hSeverity$ and $wSeverity$ are all set to low values
#         "height_severity": 1,
#         "width_severity" = 0.05,
#         "change_frequency" = 20,

# 2. Hu & Shi & Eberhart: optima modification
#     - __1__: $hSeverity = 0$ and $s \not= 0$
#         "height_severity": 0,
#         "change_severity": 1,
#     - __2__: $hSeverity \not= 0$ and $s = 0$
#         "height_severity": 1,
#         "change_severity": 0,
#     - __3__: $hSeverity \not= 0$ and $s \not= 0$
#         "height_severity": 1,
#         "change_severity": 1,
# 3. Angeline's Classification: optima trajectory
#     - __C__: $s = 0$, rotation matrix is applied to peak movement.
#         "change_severity": 0,
#         "lambda_param": 0,
#     - __L__: $\lambda = 1$ and $s \not= 0$
#         "change_severity": 1,
#         "lambda_param": 1,
#     - __R__: $\lambda = 0$ and $s \not= 0$
#         "change_severity": 1,
#         "lambda_param": 0,
# """

# A1C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A1C",
# }
# A1L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A1L",
# }
# A1R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A1R",
# }
# A2C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A2C",
# }
# A2L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A2L",
# }
# A2R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A2R",
# }
# A3C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A3C",
# }
# A3L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A3L",
# }
# A3R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 100,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "A3R",
# }
# C1C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C1C",
# }
# C1L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C1L",
# }
# C1R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C1R",
# }
# C2C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C2C",
# }
# C2L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C2L",
# }
# C2R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C2R",
# }
# C3C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C3C",
# }
# C3L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C3L",
# }
# C3R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 30,
#     "change_severity": 10,
#     "height_severity": 10,
#     "width_severity": 10,
#     "lambda_param": 0,
#     "name": "C3R",
# }
# P1C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P1C",
# }
# P1L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P1L",
# }
# P1R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P1R",
# }
# P2C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P2C",
# }
# P2L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P2L",
# }
# P2R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P2R",
# }
# P3C_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P3C",
# }
# P3L_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P3L",
# }
# P3R_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 20,
#     "change_severity": 1,
#     "height_severity": 1,
#     "width_severity": 0.05,
#     "lambda_param": 0,
#     "name": "P3R",
# }
# STA_params = {
#     "dimension": 5,
#     "num_peaks": 10,
#     "domain": (0.0, 100.0),
#     "min_height": 30.0,
#     "max_height": 70.0,
#     "min_width": 1.0,
#     "max_width": 12.0,
#     "change_frequency": 0,
#     "change_severity": 0,
#     "height_severity": 0,
#     "width_severity": 0,
#     "lambda_param": 0,
#     "name": "STA",
# }