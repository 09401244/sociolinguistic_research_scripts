import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster G block)
df = pd.read_csv(
    "out_G_langua_choice_preprocessed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Normalize Context_ columns
# Context_* columns encode features of the immediate
# interactional context at each choice event.
# Categorical context columns are dummy-coded;
# numeric ones are min-max normalized.
context_cols = [
    c for c in df.columns
    if c.startswith('Context_')]
for col in context_cols:
    if df[col].dtype == object:
        dummies = pd.get_dummies(
            df[col], prefix=col,
            drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    else:
        df[col] = pd.to_numeric(
            df[col], errors='coerce')
        rng = df[col].max() - df[col].min()
        if rng > 0:
            df[col] = (
                (df[col] - df[col].min()) / rng)
if context_cols:
    print(f"Context columns processed: "
          f"{context_cols}")

##### 2. Assemble predictor column list
predictor_cols = []
for c in ['Year_Centered', 'Age_Centered',
          'Education_Num']:
    if c in df.columns:
        predictor_cols.append(c)
predictor_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Context_'))]
predictor_cols = [
    c for c in predictor_cols
    if c in df.columns]
print(f"Predictor columns assembled: "
      f"{predictor_cols}")

##### 3. Save processed dataset
df.to_csv(
    "out_G_langua_choice_processed.csv",
    index=False)
print("Language choice modeling data processing "
      "completed.")
