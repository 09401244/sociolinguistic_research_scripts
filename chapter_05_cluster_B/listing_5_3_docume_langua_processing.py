import pandas as pd

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster B block)
df = pd.read_csv("out_B_docume_langua_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### Extract open-ended responses for qualitative analysis
# Named separately from the main path to distinguish
# qualitative outputs across methods in the same cluster.
if 'Open_Response' in df.columns:
    open_ended = df[['Open_Response']].dropna()
    open_ended.to_csv(
        "out_B_docume_langua_open_responses.csv",
        index=False)
    print(f"Open-ended responses saved: "
          f"{len(open_ended)} records.")
else:
    print("Warning: Open_Response column not found. "
          "Skipping qualitative extraction.")

##### Save processed dataset
df.to_csv("out_B_docume_langua_processed.csv", index=False)
print("Language attitudes data processing completed.")
