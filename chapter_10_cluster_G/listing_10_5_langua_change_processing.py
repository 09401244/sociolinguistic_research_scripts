import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster G block)
df = pd.read_csv(
    "out_G_langua_change_preprocessed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")
print(f"Year range: {df['Year'].min()} to "
      f"{df['Year'].max()}")

##### 1. Validate Study_Type labels
if 'Study_Type' in df.columns:
    df['Study_Type'] = (df['Study_Type'].str.strip()
                        .str.lower())
    valid_types = {'panel', 'trend'}
    invalid = ~df['Study_Type'].isin(valid_types)
    if invalid.any():
        vals = (df.loc[invalid, 'Study_Type']
                .unique().tolist())
        print(f"Warning: {invalid.sum()} records with "
              f"unrecognized Study_Type values: {vals}")
    print(f"Study type distribution:\n"
          f"{df['Study_Type'].value_counts()}")

##### 2. Check temporal completeness for panel data
if 'Study_Type' in df.columns:
    panel = df[df['Study_Type'] == 'panel']
else:
    panel = df
if len(panel) > 0 and 'Speaker_ID' in panel.columns:
    obs_per_speaker = (panel.groupby('Speaker_ID')
                       ['Year'].nunique())
    single_obs = obs_per_speaker[
        obs_per_speaker < 2]
    if not single_obs.empty:
        print(f"Warning: {len(single_obs)} panel "
              f"speaker(s) with only one observation "
              f"time point. Longitudinal models require "
              f">= 2.")
    print(f"Panel speakers: {len(obs_per_speaker)}")
    print(f"Median observation points per speaker: "
          f"{obs_per_speaker.median():.1f}")

##### 3. Save processed dataset
df.to_csv(
    "out_G_langua_change_processed.csv", index=False)
print("Language change longitudinal tracking data "
      "processing completed.")
