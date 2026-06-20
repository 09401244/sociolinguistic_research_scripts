import pandas as pd
import numpy as np

# Load preprocessed participant dataset (output of
# Prerequisite Preprocessing Script, Listing 3.1,
# Cluster F block)
participants = pd.read_csv(
    "out_F_evalua_langua_preprocessed.csv",
    encoding='utf-8')
print(f"Participant file: {len(participants)} "
      f"participants loaded.")
print(f"Group distribution: "
      f"{participants['Group'].value_counts().to_dict()}")

# Load raw outcomes file directly (second input file;
# not produced by the preprocessing script —
# prepare this file separately before running)
outcomes = pd.read_csv(
    "intervention_outcomes_raw.csv", encoding='utf-8')
outcomes = outcomes.drop_duplicates()
outcomes = outcomes.dropna(
    subset=['Participant_ID', 'Time'])
outcomes['Participant_ID'] = (
    outcomes['Participant_ID'].astype(str).str.strip())
print(f"Outcomes file: {len(outcomes)} records loaded.")

##### 1. Validate Implementation_Fidelity range
if 'Implementation_Fidelity' in participants.columns:
    participants['Implementation_Fidelity'] = (
        pd.to_numeric(
            participants['Implementation_Fidelity'],
            errors='coerce'))
    invalid_fid = (
        participants['Implementation_Fidelity'].notna()
        & ((participants['Implementation_Fidelity'] < 0)
           | (participants['Implementation_Fidelity']
              > 100)))
    if invalid_fid.any():
        print(f"Warning: {invalid_fid.sum()} records "
              f"with Implementation_Fidelity outside "
              f"0-100 range.")

##### 2. Validate Time labels in outcomes file
outcomes['Time'] = (outcomes['Time'].str.strip()
                    .str.lower())
valid_times = {'pre', 'post'}
invalid_times = ~outcomes['Time'].isin(valid_times)
if invalid_times.any():
    print(f"Warning: {invalid_times.sum()} outcome "
          f"records with unrecognized Time values. "
          f"Removing.")
    outcomes = outcomes[~invalid_times]

# Flag duplicate Participant_ID + Time combinations
dupes = outcomes.duplicated(
    subset=['Participant_ID', 'Time'])
if dupes.any():
    print(f"Warning: {dupes.sum()} duplicate "
          f"Participant_ID + Time combinations. "
          f"Keeping first occurrence.")
    outcomes = outcomes.drop_duplicates(
        subset=['Participant_ID', 'Time'], keep='first')

##### 3. Validate outcome score ranges
if 'Attitude_Score' in outcomes.columns:
    outcomes['Attitude_Score'] = pd.to_numeric(
        outcomes['Attitude_Score'], errors='coerce')
    invalid_att = (
        outcomes['Attitude_Score'].notna()
        & ((outcomes['Attitude_Score'] < 1)
           | (outcomes['Attitude_Score'] > 5)))
    if invalid_att.any():
        print(f"Warning: {invalid_att.sum()} "
              f"Attitude_Score values outside expected "
              f"range 1--5.")

if 'Language_Use_Score' in outcomes.columns:
    outcomes['Language_Use_Score'] = pd.to_numeric(
        outcomes['Language_Use_Score'], errors='coerce')
    invalid_use = (
        outcomes['Language_Use_Score'].notna()
        & ((outcomes['Language_Use_Score'] < 0)
           | (outcomes['Language_Use_Score'] > 100)))
    if invalid_use.any():
        print(f"Warning: {invalid_use.sum()} "
              f"Language_Use_Score values outside "
              f"range 0--100.")

##### 4. Check temporal completeness
# Each participant should have both pre and post records.
time_counts = (
    outcomes.groupby('Participant_ID')['Time']
    .apply(set).reset_index())
time_counts.columns = ['Participant_ID', 'Time_Points']
missing_pre = time_counts[
    time_counts['Time_Points'].apply(
        lambda x: 'pre' not in x)]
missing_post = time_counts[
    time_counts['Time_Points'].apply(
        lambda x: 'post' not in x)]
if not missing_pre.empty:
    print(f"Warning: {len(missing_pre)} participant(s) "
          f"missing pre-intervention assessment.")
if not missing_post.empty:
    print(f"Warning: {len(missing_post)} participant(s) "
          f"missing post-intervention assessment.")

##### 5. Impute missing outcome scores
# Group-time mean imputation preserves the pre-post
# structure and avoids collapsing group differences.
outcome_cols = [
    c for c in ['Attitude_Score', 'Language_Use_Score']
    if c in outcomes.columns]
outcomes = outcomes.merge(
    participants[['Participant_ID', 'Group']],
    on='Participant_ID', how='left')
for col in outcome_cols:
    group_time_means = (
        outcomes.groupby(['Group', 'Time'])[col]
        .transform('mean'))
    n_imputed = outcomes[col].isna().sum()
    outcomes[col] = outcomes[col].fillna(
        group_time_means)
    if n_imputed > 0:
        print(f"{col}: {n_imputed} missing values "
              f"imputed with group-time mean.")

##### 6. Extract and preserve open-ended responses
if 'Open_Response' in outcomes.columns:
    open_responses = (
        outcomes[['Participant_ID', 'Time',
                  'Open_Response']]
        .dropna(subset=['Open_Response']))
    open_responses = open_responses[
        open_responses['Open_Response']
        .str.strip().str.len() > 0]
    open_responses.to_csv(
        "out_F_evalua_langua_open_responses.csv",
        index=False)
    print(f"Open responses preserved: "
          f"{len(open_responses)} records.")
    outcomes = outcomes.drop(columns=['Open_Response'])

##### 7. Reshape to wide format
# (one row per participant)
wide = outcomes.pivot(
    index='Participant_ID',
    columns='Time',
    values=outcome_cols)
wide.columns = [
    f"{col}_{time}" for col, time in wide.columns]
wide = wide.reset_index()

# Compute change scores (post minus pre) per outcome
for col in outcome_cols:
    pre_col  = f"{col}_pre"
    post_col = f"{col}_post"
    if (pre_col in wide.columns
            and post_col in wide.columns):
        wide[f"{col}_Change"] = (
            wide[post_col] - wide[pre_col])

##### 8. Merge participant attributes onto wide data
df_final = wide.merge(
    participants.drop(columns=['Group'],
                      errors='ignore'),
    on='Participant_ID', how='left')
df_final = df_final.merge(
    participants[['Participant_ID', 'Group']],
    on='Participant_ID', how='left')

##### 9. Save outputs
df_final.to_csv(
    "out_F_evalua_langua_processed.csv", index=False)
print("Evaluating language interventions data processing "
      "completed.")
print("Output files:")
print("  out_F_evalua_langua_processed.csv      "
      "-- wide-format dataset with pre, post, and "
      "change scores")
print("  out_F_evalua_langua_open_responses.csv "
      "-- qualitative open responses")
