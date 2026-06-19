import pandas as pd
from collections import Counter
import re

# Load processed survey data (output of Listing 5.3)
df = pd.read_csv("out_B_docume_langua_processed.csv")

##### Descriptive statistics for Likert-scale items
likert_cols = [c for c in ['Attitude_Q1', 'Attitude_Q2']
               if c in df.columns]
if likert_cols:
    print("Descriptive Statistics:")
    print(df[likert_cols].describe())

    ##### Mean attitude score
    df['Mean_Attitude_Score'] = df[likert_cols].mean(axis=1)
    print("\nOverall Mean Attitude Score:")
    print(df['Mean_Attitude_Score'].mean())
else:
    print("Warning: No Attitude_Q columns found. "
          "Skipping descriptive statistics.")

##### Load open-ended responses
open_file = "out_B_docume_langua_open_responses.csv"
try:
    open_df = pd.read_csv(open_file)

    ##### Basic word frequency analysis
    if 'Open_Response' in open_df.columns:
        all_text = " ".join(
            open_df['Open_Response']
            .dropna().astype(str))
        all_text = re.sub(
            r'[^\w\s]', '', all_text.lower())
        words = all_text.split()
        word_counts = Counter(words)
        print("\nMost Common Words in Open-Ended "
              "Responses:")
        for word, count in word_counts.most_common(15):
            print(f"  {word}: {count}")
    else:
        print("Warning: Open_Response column not found "
              f"in {open_file}. Skipping word frequency.")
except FileNotFoundError:
    print(f"Warning: {open_file} not found. "
          "Skipping open-ended response analysis. "
          "Run Listing 5.3 first to generate this file.")

print("\nLanguage attitudes analysis completed.")
