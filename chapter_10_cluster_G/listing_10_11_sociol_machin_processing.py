import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster G block)
df = pd.read_csv(
    "out_G_sociol_machin_preprocessed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Community dummy coding
if 'Community' in df.columns:
    comm_dummies = pd.get_dummies(
        df['Community'], prefix='Comm',
        drop_first=True, dtype=int)
    df = pd.concat([df, comm_dummies], axis=1)
    print(f"Community dummies created: "
          f"{comm_dummies.columns.tolist()}")

##### 2. Normalize additional Feature_ columns
# Any columns named Feature_* are treated as
# supplementary numeric predictors and min-max
# normalized.
feature_extra = [
    c for c in df.columns
    if c.startswith('Feature_')]
for col in feature_extra:
    df[col] = pd.to_numeric(
        df[col], errors='coerce')
    rng = df[col].max() - df[col].min()
    if rng > 0:
        df[col] = (
            (df[col] - df[col].min()) / rng)
if feature_extra:
    print(f"Feature_ columns normalized: "
          f"{feature_extra}")

##### 3. Assemble feature matrix and validity report
feature_cols = ['Year_Centered']
for c in ['Age_Centered', 'Education_Num']:
    if c in df.columns:
        feature_cols.append(c)
feature_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Comm_',
                     'Feature_'))]
feature_cols = [
    c for c in feature_cols if c in df.columns]
print(f"Feature columns assembled "
      f"({len(feature_cols)}): {feature_cols}")
missing_in_features = (df[feature_cols]
                       .isna().sum())
if missing_in_features.any():
    print(f"Warning: missing values in feature "
          f"matrix:\n"
          f"{missing_in_features[missing_in_features > 0]}")

##### 4. Save modeling-ready dataset
df.to_csv(
    "out_G_sociol_machin_processed.csv",
    index=False)
print("Sociolinguistic machine learning data "
      "processing completed.")
