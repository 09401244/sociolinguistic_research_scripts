import pandas as pd
import numpy as np

# Load simulation parameter file
# (primary input for this method — not produced
# by the preprocessing script; prepare separately)
params_df = pd.read_csv(
    "simulation_params.csv", encoding='utf-8')
print(f"Parameter configurations loaded: "
      f"{len(params_df)}")

##### 1. Validate required parameter columns
required_params = [
    'N_Agents', 'N_Steps', 'Init_Freq_A',
    'Prestige_Bias', 'Network_Density',
    'N_Simulations']
missing = [
    c for c in required_params
    if c not in params_df.columns]
if missing:
    raise ValueError(
        f"Missing parameter columns: {missing}")

##### 2. Coerce all parameters to numeric
for col in required_params:
    params_df[col] = pd.to_numeric(
        params_df[col], errors='coerce')
    invalid = params_df[col].isna()
    if invalid.any():
        print(f"Warning: {invalid.sum()} non-numeric "
              f"values in {col} (set to NaN).")

##### 3. Validate parameter ranges
range_checks = {
    'N_Agents':        (1,    None),
    'N_Steps':         (1,    None),
    'Init_Freq_A':     (0.0,  1.0),
    'Prestige_Bias':   (-1.0, 1.0),
    'Network_Density': (0.0,  1.0),
    'N_Simulations':   (1,    None),
}
for col, (lo, hi) in range_checks.items():
    if col not in params_df.columns:
        continue
    mask = params_df[col].notna()
    if lo is not None:
        invalid_lo = mask & (params_df[col] < lo)
        if invalid_lo.any():
            print(f"Warning: {invalid_lo.sum()} rows "
                  f"with {col} < {lo}.")
    if hi is not None:
        invalid_hi = mask & (params_df[col] > hi)
        if invalid_hi.any():
            print(f"Warning: {invalid_hi.sum()} rows "
                  f"with {col} > {hi}.")

##### 4. Drop rows with any invalid parameter values
before = len(params_df)
params_df = params_df.dropna(subset=required_params)
removed = before - len(params_df)
if removed > 0:
    print(f"{removed} parameter rows removed due to "
          f"invalid values.")

##### 5. Save validated parameter file
params_df.to_csv(
    "out_G_comput_simula_preprocessed.csv",
    index=False)
print(f"Validated parameter configurations: "
      f"{len(params_df)}")
print("Computational simulation data processing "
      "completed.")
