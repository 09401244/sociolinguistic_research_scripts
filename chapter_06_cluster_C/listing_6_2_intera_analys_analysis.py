import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Load processed and coded dataset (output of Listing 6.1)
df = pd.read_csv("out_C_intera_analys_processed.csv",
                 encoding='utf-8')

##### 1. Turn-taking: frequency distribution of turn types
turn_by_speaker = None
if 'Turn_Type' in df.columns:
    print("Frequency Distribution of Turn Types:")
    turn_counts = df['Turn_Type'].value_counts()
    print(turn_counts)
    turn_counts.plot(kind='bar')
    plt.title("Distribution of Turn Types")
    plt.xlabel("Turn Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    if 'Speaker' in df.columns:
        turn_by_speaker = pd.crosstab(
            df['Speaker'], df['Turn_Type'])
        print("\nTurn Type by Speaker:")
        print(turn_by_speaker)
    else:
        print("Warning: Speaker column not found. "
              "Skipping turn type by speaker crosstab.")
else:
    print("Warning: Turn_Type column not found. "
          "Skipping step 1.")

##### 2. Turn-taking: overlap and gap distributions
if 'Overlap' in df.columns:
    print("\nOverlap Distribution:")
    overlap_counts = df['Overlap'].value_counts()
    print(overlap_counts)
    overlap_counts.plot(
        kind='pie', autopct='%1.1f%%',
        labels=['No Overlap', 'Overlap'])
    plt.title("Distribution of Overlap")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Overlap column not found. "
          "Skipping overlap distribution.")

if 'Gap' in df.columns:
    print("\nGap Descriptive Statistics (seconds):")
    print(df['Gap'].describe())
    df['Gap'].dropna().hist(bins=20)
    plt.title("Distribution of Inter-Turn Gap (seconds)")
    plt.xlabel("Gap (seconds)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Gap column not found. "
          "Skipping gap distribution.")

##### 3. Sequence organization: position distribution
if 'Sequence_Position' in df.columns:
    print("\nFrequency Distribution of Sequence Positions:")
    seq_counts = df['Sequence_Position'].value_counts()
    print(seq_counts)
    seq_counts.plot(kind='bar')
    plt.title("Distribution of Sequence Positions")
    plt.xlabel("Sequence Position")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Sequence_Position column not found. "
          "Skipping step 3.")

##### 4. Sequence organization: pair type distribution
pos_by_pair = None
if 'Pair_Type' in df.columns:
    print("\nFrequency Distribution of Adjacency "
          "Pair Types:")
    pair_counts = df['Pair_Type'].value_counts()
    print(pair_counts)
    pair_counts.plot(kind='bar')
    plt.title("Distribution of Adjacency Pair Types")
    plt.xlabel("Pair Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    if 'Sequence_Position' in df.columns:
        pos_by_pair = pd.crosstab(
            df['Pair_Type'], df['Sequence_Position'])
        print("\nSequence Position by Pair Type:")
        print(pos_by_pair)
    else:
        print("Warning: Sequence_Position column not "
              "found. Skipping pair type crosstab.")
else:
    print("Warning: Pair_Type column not found. "
          "Skipping step 4.")

##### 5. Repair: initiation and completion distributions
repair_crosstab = None
repair_by_speaker = None
if 'Repair_Initiation' in df.columns:
    print("\nRepair Initiation Distribution:")
    repair_init_counts = (df['Repair_Initiation']
                          .value_counts())
    print(repair_init_counts)
    repair_init_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Repair Initiation")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Repair_Initiation column not found. "
          "Skipping repair initiation distribution.")

if 'Repair_Completion' in df.columns:
    print("\nRepair Completion Distribution:")
    repair_comp_counts = (df['Repair_Completion']
                          .value_counts())
    print(repair_comp_counts)
    repair_comp_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Repair Completion")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    if ('Speaker' in df.columns
            and 'Repair_Initiation' in df.columns):
        repair_by_speaker = pd.crosstab(
            df['Speaker'], df['Repair_Initiation'])
        print("\nRepair Initiation by Speaker:")
        print(repair_by_speaker)
    if 'Repair_Initiation' in df.columns:
        repair_crosstab = pd.crosstab(
            df['Repair_Initiation'],
            df['Repair_Completion'])
        print("\nRepair Initiation by Repair Completion:")
        print(repair_crosstab)
else:
    print("Warning: Repair_Completion column not found. "
          "Skipping repair completion distribution.")

# Chi-square: repair initiation vs. completion
if repair_crosstab is not None:
    chi2, p_val, dof, expected = stats.chi2_contingency(
        repair_crosstab)
    print(f"\nChi-Square Test: Repair Initiation vs. "
          f"Completion:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")
else:
    print("Warning: Chi-square test (Repair) skipped — "
          "crosstab from step 5 was not computed.")

##### 6. Participant alignment: distribution
align_by_pair = None
if 'Alignment' in df.columns:
    print("\nParticipant Alignment Distribution:")
    align_counts = df['Alignment'].value_counts()
    print(align_counts)
    align_counts.plot(kind='bar')
    plt.title("Distribution of Participant Alignment")
    plt.xlabel("Alignment")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    if 'Pair_Type' in df.columns:
        align_by_pair = pd.crosstab(
            df['Pair_Type'], df['Alignment'])
        print("\nAlignment by Pair Type:")
        print(align_by_pair)
    else:
        print("Warning: Pair_Type column not found. "
              "Skipping alignment by pair type crosstab.")
    if 'Speaker' in df.columns:
        align_by_speaker = pd.crosstab(
            df['Speaker'], df['Alignment'])
        print("\nAlignment by Speaker:")
        print(align_by_speaker)
    else:
        print("Warning: Speaker column not found. "
              "Skipping alignment by speaker crosstab.")
else:
    print("Warning: Alignment column not found. "
          "Skipping step 6.")

# Chi-square: pair type vs. alignment
if align_by_pair is not None:
    chi2, p_val, dof, expected = stats.chi2_contingency(
        align_by_pair)
    print(f"\nChi-Square Test: Pair Type vs. Alignment:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")
else:
    print("Warning: Chi-square test (Alignment) skipped — "
          "crosstab from step 6 was not computed.")

##### 7. Turn duration and length analysis
if 'Turn_Duration' in df.columns:
    if 'Speaker' in df.columns:
        print("\nMean Turn Duration by Speaker (seconds):")
        print(df.groupby('Speaker')
              ['Turn_Duration'].mean())
    else:
        print("Warning: Speaker column not found. "
              "Skipping turn duration by speaker.")
    if 'Turn_Type' in df.columns:
        print("\nMean Turn Duration by Turn Type "
              "(seconds):")
        print(df.groupby('Turn_Type')
              ['Turn_Duration'].mean())
    else:
        print("Warning: Turn_Type column not found. "
              "Skipping turn duration by type.")
else:
    print("Warning: Turn_Duration column not found. "
          "Skipping step 7a.")

if 'Turn_Length' in df.columns:
    if 'Speaker' in df.columns:
        print("\nMean Turn Length by Speaker (tokens):")
        print(df.groupby('Speaker')
              ['Turn_Length'].mean())
    else:
        print("Warning: Speaker column not found. "
              "Skipping turn length by speaker.")
else:
    print("Warning: Turn_Length column not found. "
          "Skipping step 7b.")

##### 8. Transition-relevance place analysis
if 'TRP_Flagged' in df.columns:
    if 'Turn_Type' in df.columns:
        trp_by_turn = pd.crosstab(
            df['Turn_Type'], df['TRP_Flagged'])
        print("\nTRP Distribution by Turn Type:")
        print(trp_by_turn)
    else:
        print("Warning: Turn_Type column not found. "
              "Skipping TRP by turn type.")
else:
    print("Warning: TRP_Flagged column not found. "
          "Skipping step 8.")

##### 9. Co-occurrence analysis of coding categories
code_columns = [col for col in df.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\nCo-occurrence Counts Among Coding Categories:")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (df[col_a] == 1) & (df[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")
else:
    print("Warning: Fewer than 2 Code_ columns found. "
          "Skipping co-occurrence analysis.")

print("Interactional analysis completed.")
