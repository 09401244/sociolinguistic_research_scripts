import pandas as pd
import numpy as np
import re

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster E block)
df = pd.read_csv("out_E_sociol_corpus_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Validate genre labels against controlled vocabulary
if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].str.strip().str.lower()
    valid_genres = {
        'spoken', 'written', 'digital', 'academic',
        'informal', 'institutional', 'literary'}
    unrecognized = ~df['Genre'].isin(valid_genres)
    if unrecognized.any():
        vals = (df.loc[unrecognized, 'Genre']
                .unique().tolist())
        print(f"Warning: {unrecognized.sum()} records "
              f"with unrecognized Genre values: {vals}")

##### 2. Compute corpus statistics per text
content_col = 'Text_Content'
df['Token_Count'] = df[content_col].apply(
    lambda x: len(str(x).split()))
df['Type_Count'] = df[content_col].apply(
    lambda x: len(set(str(x).lower().split())))
df['TTR'] = np.where(
    df['Token_Count'] > 0,
    df['Type_Count'] / df['Token_Count'],
    np.nan)
df['Sentence_Count'] = df[content_col].apply(
    lambda x: max(1, len(
        re.findall(r'[.!?]+', str(x)))))
df['Mean_Sentence_Length'] = np.where(
    df['Sentence_Count'] > 0,
    df['Token_Count'] / df['Sentence_Count'],
    np.nan)
total_tokens = df['Token_Count'].sum()
mean_tokens = df['Token_Count'].mean()
print(f"Corpus statistics computed for {len(df)} texts.")
print(f"Total tokens: {total_tokens:,}")
print(f"Mean tokens per text: {mean_tokens:.1f}")

##### 3. Initialize binary Code_ columns for
# linguistic feature annotation.
# Left as NaN for manual analyst coding.
# Adapt feature names to the research question.
code_columns = [
    'Code_Lexical_Innovation',
    'Code_Code_Switch',
    'Code_Register_Shift',
    'Code_Stance_Marker',
    'Code_Identity_Reference']
for col in code_columns:
    df[col] = np.nan

##### 4. Build and export corpus summary statistics
all_types = (df[content_col]
             .apply(lambda x: set(str(x).lower().split()))
             .apply(list))
total_types = (len(set(
    t for sublist in all_types for t in sublist))
    if len(df) > 0 else 0)
summary = {
    'Total_Texts': len(df),
    'Total_Tokens': int(total_tokens),
    'Total_Types': total_types,
    'Mean_Tokens_Per_Text': round(mean_tokens, 2),
    'Mean_TTR': round(df['TTR'].mean(), 4),
}
if 'Speaker_ID' in df.columns:
    summary['Unique_Speakers'] = int(
        df['Speaker_ID'].nunique())
if 'Genre' in df.columns:
    summary['Unique_Genres'] = int(
        df['Genre'].nunique())
if 'Year' in df.columns and df['Year'].notna().any():
    summary['Year_Range'] = (
        f"{int(df['Year'].min())}--"
        f"{int(df['Year'].max())}")
summary_df = pd.DataFrame([summary])
summary_df.to_csv(
    "out_E_sociol_corpus_summary.csv", index=False)

##### 5. Save processed corpus
df.to_csv("out_E_sociol_corpus_processed.csv",
          index=False)
print("Sociolinguistic corpus analysis data processing "
      "completed.")
print("Output files:")
print("  out_E_sociol_corpus_processed.csv  "
      "-- cleaned corpus with statistics and "
      "initialized Code_ columns")
print("  out_E_sociol_corpus_summary.csv    "
      "-- aggregate corpus statistics")
