import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Load processed dataset (output of Listing 4.3)
df = pd.read_csv("out_A_rhetor_strate_processed.csv")

##### 1. Frequency distribution of argumentation schemes
print("Frequency Distribution of Argumentation Schemes:")
scheme_counts = df['Argumentation_Scheme'].value_counts()
print(scheme_counts)

# Bar chart of scheme frequencies
scheme_counts.plot(kind='bar')
plt.title("Distribution of Argumentation Schemes")
plt.xlabel("Argumentation Scheme")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

##### 2. Critical questions status distribution
if 'Critical_Questions_Status' in df.columns:
    print("\nCritical Questions Status Distribution:")
    cq_counts = (df['Critical_Questions_Status']
                 .value_counts())
    print(cq_counts)
    cq_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Critical Questions Status")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    # Critical questions status by argumentation scheme
    if 'Argumentation_Scheme' in df.columns:
        cq_by_scheme = pd.crosstab(
            df['Argumentation_Scheme'],
            df['Critical_Questions_Status'])
        print("\nCritical Questions Status by "
              "Argumentation Scheme:")
        print(cq_by_scheme)

##### 3. Frequency distribution of rhetorical devices
if 'Rhetorical_Device' in df.columns:
    print("\nFrequency Distribution of Rhetorical Devices:")
    device_counts = df['Rhetorical_Device'].value_counts()
    print(device_counts)
    device_counts.plot(kind='bar')
    plt.title("Distribution of Rhetorical Devices")
    plt.xlabel("Rhetorical Device")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

##### 4. Device type distribution
if 'Device_Type' in df.columns:
    print("\nDevice Type Distribution:")
    dtype_counts = df['Device_Type'].value_counts()
    print(dtype_counts)
    dtype_counts.plot(kind='bar')
    plt.title("Distribution of Device Types")
    plt.xlabel("Device Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # Device type by rhetorical device
    if 'Rhetorical_Device' in df.columns:
        dtype_by_device = pd.crosstab(
            df['Device_Type'],
            df['Rhetorical_Device'])
        print("\nDevice Type by Rhetorical Device:")
        print(dtype_by_device)

##### 5. Audience positioning distribution
if 'Audience_Positioning' in df.columns:
    print("\nAudience Positioning Distribution:")
    ap_counts = df['Audience_Positioning'].value_counts()
    print(ap_counts)
    ap_counts.plot(kind='bar')
    plt.title("Distribution of Audience Positioning "
              "Strategies")
    plt.xlabel("Audience Positioning")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # Audience positioning by argumentation scheme
    if 'Argumentation_Scheme' in df.columns:
        ap_by_scheme = pd.crosstab(
            df['Argumentation_Scheme'],
            df['Audience_Positioning'])
        print("\nAudience Positioning by "
              "Argumentation Scheme:")
        print(ap_by_scheme)

##### 6. Evaluative framing distribution
if 'Evaluative_Framing' in df.columns:
    print("\nEvaluative Framing Distribution:")
    framing_counts = (df['Evaluative_Framing']
                      .value_counts())
    print(framing_counts)
    framing_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Evaluative Framing")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    # Evaluative framing by rhetorical device
    if 'Rhetorical_Device' in df.columns:
        framing_by_device = pd.crosstab(
            df['Rhetorical_Device'],
            df['Evaluative_Framing'])
        print("\nEvaluative Framing by Rhetorical Device:")
        print(framing_by_device)

##### 7. Chi-square test: argumentation scheme vs.
# critical questions status
if ('Argumentation_Scheme' in df.columns
        and 'Critical_Questions_Status' in df.columns):
    chi2, p_val, dof, expected = stats.chi2_contingency(
        cq_by_scheme)
    print(f"\nChi-Square Test: Argumentation Scheme vs. "
          f"Critical Questions Status:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")

##### 8. Chi-square test: rhetorical device vs.
# evaluative framing
if ('Rhetorical_Device' in df.columns
        and 'Evaluative_Framing' in df.columns):
    chi2, p_val, dof, expected = stats.chi2_contingency(
        framing_by_device)
    print(f"\nChi-Square Test: Rhetorical Device vs. "
          f"Evaluative Framing:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")

##### 9. Co-occurrence analysis of coding categories
code_columns = [col for col in df.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\nCo-occurrence Counts Among Coding Categories:")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (df[col_a] == 1) & (df[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")

print("Rhetorical strategy analysis completed.")
