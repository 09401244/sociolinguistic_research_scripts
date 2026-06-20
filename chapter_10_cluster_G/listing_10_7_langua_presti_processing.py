import pandas as pd
import numpy as np

# Load raw prestige dataset directly.
# This method uses a domain-level longitudinal file
# rather than a speaker-level corpus; it does not
# pass through the standard Cluster G preprocessing
# block. Prepare prestige_raw.csv separately.
df = pd.read_csv("prestige_raw.csv", encoding='utf-8')
df = df.drop_duplicates()
print(f"Records loaded: {len(df)}")

##### 1. Validate required columns
required = ['Year', 'Domain', 'Language',
            'Corpus_Share']
missing = [c for c in required
           if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}")
df = df.dropna(subset=required)

##### 2. Normalize Domain and Language fields
df['Domain']   = (df['Domain'].str.strip()
                  .str.lower())
df['Language'] = (df['Language'].str.strip()
                  .str.title())

##### 3. Coerce and validate numeric columns
df['Year'] = pd.to_numeric(
    df['Year'], errors='coerce')
df['Corpus_Share'] = pd.to_numeric(
    df['Corpus_Share'], errors='coerce')
invalid_cs = (
    df['Corpus_Share'].notna()
    & ((df['Corpus_Share'] < 0)
       | (df['Corpus_Share'] > 1)))
if invalid_cs.any():
    print(f"Warning: {invalid_cs.sum()} "
          f"Corpus_Share values outside range 0--1.")

if 'Attitude_Score' in df.columns:
    df['Attitude_Score'] = pd.to_numeric(
        df['Attitude_Score'], errors='coerce')
    invalid_att = (
        df['Attitude_Score'].notna()
        & ((df['Attitude_Score'] < 1)
           | (df['Attitude_Score'] > 5)))
    if invalid_att.any():
        print(f"Warning: {invalid_att.sum()} "
              f"Attitude_Score values outside "
              f"range 1--5.")

if 'Speaker_Count' in df.columns:
    df['Speaker_Count'] = pd.to_numeric(
        df['Speaker_Count'], errors='coerce')

##### 4. Compute composite prestige index per record
# Index combines normalized Corpus_Share and
# Attitude_Score (when available). Both components
# are min-max normalized before combining.
def minmax(series):
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index)
    return (series - series.min()) / rng

df['Corpus_Share_Norm'] = minmax(df['Corpus_Share'])
if 'Attitude_Score' in df.columns:
    df['Attitude_Norm'] = minmax(
        df['Attitude_Score'].fillna(
            df['Attitude_Score'].mean()))
    df['Prestige_Index'] = (
        (df['Corpus_Share_Norm']
         + df['Attitude_Norm']) / 2)
else:
    df['Prestige_Index'] = df['Corpus_Share_Norm']

# Aggregate prestige index per Year
prestige_by_year = (
    df.groupby('Year')['Prestige_Index']
    .mean().reset_index()
    .rename(columns={
        'Prestige_Index': 'Mean_Prestige_Index'}))
print(f"Prestige index computed for "
      f"{len(prestige_by_year)} years.")

##### 5. Save processed outputs
df.to_csv(
    "out_G_langua_presti_processed.csv", index=False)
prestige_by_year.to_csv(
    "out_G_langua_presti_by_year.csv", index=False)
print("Language prestige analysis data processing "
      "completed.")
print("Output files:")
print("  out_G_langua_presti_processed.csv  "
      "-- record-level data with prestige index")
print("  out_G_langua_presti_by_year.csv    "
      "-- annual mean prestige index for trend "
      "modeling")
