import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster G block)
df = pd.read_csv(
    "out_G_predic_sociol_preprocessed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")
print(f"Year range: {df['Year'].min()} to "
      f"{df['Year'].max()}")

##### 1. Community dummy coding (if cardinality permits)
if 'Community' in df.columns:
    n_comm = df['Community'].nunique()
    if n_comm <= 10:
        comm_dummies = pd.get_dummies(
            df['Community'], prefix='Comm',
            drop_first=True, dtype=int)
        df = pd.concat([df, comm_dummies], axis=1)
        print(f"Community dummies created "
              f"({n_comm} categories): "
              f"{comm_dummies.columns.tolist()}")
    else:
        print(f"Warning: {n_comm} communities detected; "
              f"dummy coding skipped. Consider grouping "
              f"or using a random-effects term instead.")

##### 2. Assemble and record feature column list
feature_cols = ['Year_Centered']
for c in ['Age_Centered', 'Education_Num']:
    if c in df.columns:
        feature_cols.append(c)
feature_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Comm_'))]
feature_cols = [
    c for c in feature_cols if c in df.columns]
print(f"Feature columns assembled: {feature_cols}")

##### 3. Save modeling-ready dataset
df.to_csv(
    "out_G_predic_sociol_processed.csv", index=False)
print("Predictive sociolinguistic modeling data "
      "processing completed.")
