import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster A block)
df = pd.read_csv("out_A_rhetor_strate_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Initialize rhetorical coding columns
# These columns are left as NaN for manual analyst coding.

# Argumentation scheme dimension
df['Argumentation_Scheme'] = np.nan
# e.g., expert opinion, analogy, consequences,
#        sign, popular opinion

# Critical questions status dimension
df['Critical_Questions_Status'] = np.nan
# e.g., raised, answered, suppressed, deflected

# Rhetorical device dimension
df['Rhetorical_Device'] = np.nan
# e.g., metaphor, analogy, personification, presupposition

# Device type dimension
df['Device_Type'] = np.nan
# e.g., figurative, structural, evaluative

# Audience positioning dimension
df['Audience_Positioning'] = np.nan
# e.g., ethos, pathos, logos

# Evaluative framing dimension
df['Evaluative_Framing'] = np.nan
# e.g., positive, negative, neutral

##### 2. Initialize binary Code_ columns for
# co-occurrence analysis (adapt names to research context)
for col in ['Code_Scheme_Active', 'Code_CQ_Suppressed',
            'Code_Metaphor', 'Code_Ethos_Appeal',
            'Code_Negative_Framing']:
    df[col] = np.nan

##### 3. Save coding-ready dataset
df = df.drop(columns=['Analytical_Category',
             'Code_Feature_A', 'Code_Feature_B'],
             errors='ignore')
df.to_csv("out_A_rhetor_strate_processed.csv", index=False)
print(f"Records saved: {len(df)}")
print("Rhetorical strategy analysis data processing "
      "completed.")
