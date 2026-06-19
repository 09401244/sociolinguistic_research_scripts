import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Load processed and coded dataset (output of Listing 4.5)
df = pd.read_csv("out_A_inclus_langua_processed.csv",
                 encoding='utf-8')

##### 1. Frequency distribution: nomination and categorization
if 'Nomination_Category' in df.columns:
    print("Frequency Distribution of Nomination Categories:")
    nom_counts = df['Nomination_Category'].value_counts()
    print(nom_counts)
    nom_counts.plot(kind='bar')
    plt.title("Distribution of Nomination Categories")
    plt.xlabel("Nomination Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

##### 2. Frequency distribution: representation and salience
if 'Salience' in df.columns:
    print("\nFrequency Distribution of Salience Codes:")
    salience_counts = df['Salience'].value_counts()
    print(salience_counts)
    salience_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Representational Salience")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    # Salience by social actor
    if 'Social_Actor' in df.columns:
        salience_by_actor = pd.crosstab(
            df['Social_Actor'], df['Salience'])
        print("\nSalience by Social Actor:")
        print(salience_by_actor)

##### 3. Frequency distribution: normalization and markedness
if 'Markedness' in df.columns:
    print("\nFrequency Distribution of Markedness Codes:")
    marked_counts = df['Markedness'].value_counts()
    print(marked_counts)
    marked_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Markedness")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    # Markedness by social actor
    if 'Social_Actor' in df.columns:
        markedness_by_actor = pd.crosstab(
            df['Social_Actor'], df['Markedness'])
        print("\nMarkedness by Social Actor:")
        print(markedness_by_actor)

##### 4. Cross-tabulation: nomination category by genre
if ('Nomination_Category' in df.columns
        and 'Genre' in df.columns):
    nom_by_genre = pd.crosstab(
        df['Genre'], df['Nomination_Category'])
    print("\nNomination Category by Genre:")
    print(nom_by_genre)
    # Heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(nom_by_genre, aspect='auto', cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xticks(range(len(nom_by_genre.columns)),
               nom_by_genre.columns,
               rotation=45, ha='right')
    plt.yticks(range(len(nom_by_genre.index)),
               nom_by_genre.index)
    plt.title("Heatmap: Nomination Category by Genre")
    plt.tight_layout()
    plt.show()

##### 5. Cross-tabulation: salience by genre
if 'Salience' in df.columns and 'Genre' in df.columns:
    salience_by_genre = pd.crosstab(
        df['Genre'], df['Salience'])
    print("\nSalience by Genre:")
    print(salience_by_genre)

##### 6. Cross-tabulation: markedness by genre
if 'Markedness' in df.columns and 'Genre' in df.columns:
    markedness_by_genre = pd.crosstab(
        df['Genre'], df['Markedness'])
    print("\nMarkedness by Genre:")
    print(markedness_by_genre)

##### 7. Chi-square tests of association
# Nomination category vs. genre
if ('Nomination_Category' in df.columns
        and 'Genre' in df.columns):
    chi2, p_val, dof, expected = stats.chi2_contingency(
        nom_by_genre)
    print(f"\nChi-Square Test: Nomination Category "
          f"vs. Genre:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")
# Salience vs. social actor
if ('Salience' in df.columns
        and 'Social_Actor' in df.columns):
    chi2, p_val, dof, expected = stats.chi2_contingency(
        salience_by_actor)
    print(f"\nChi-Square Test: Salience vs. Social Actor:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")
# Markedness vs. social actor
if ('Markedness' in df.columns
        and 'Social_Actor' in df.columns):
    chi2, p_val, dof, expected = stats.chi2_contingency(
        markedness_by_actor)
    print(f"\nChi-Square Test: Markedness vs. "
          f"Social Actor:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")

##### 8. Flagged unit analysis by genre and social actor
if 'Flagged' in df.columns:
    print("\nFlagged Unit Distribution by Genre:")
    flagged_by_genre = (
        df[df['Flagged'] == True]
        .groupby('Genre')['Unit_ID'].count())
    print(flagged_by_genre)
    flagged_by_genre.plot(kind='bar')
    plt.title("Flagged Units by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    if 'Social_Actor' in df.columns:
        print("\nFlagged Unit Distribution by Social Actor:")
        flagged_by_actor = (
            df[df['Flagged'] == True]
            .groupby('Social_Actor')['Unit_ID'].count())
        print(flagged_by_actor)
        flagged_by_actor.plot(kind='bar')
        plt.title("Flagged Units by Social Actor")
        plt.xlabel("Social Actor")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

##### 9. Unit length analysis by audit axis
if 'Unit_Length' in df.columns:
    if 'Nomination_Category' in df.columns:
        print("\nMean Unit Length by Nomination Category:")
        print(df.groupby('Nomination_Category')
              ['Unit_Length'].mean())
    if 'Salience' in df.columns:
        print("\nMean Unit Length by Salience:")
        print(df.groupby('Salience')['Unit_Length'].mean())
    if 'Markedness' in df.columns:
        print("\nMean Unit Length by Markedness:")
        print(df.groupby('Markedness')
              ['Unit_Length'].mean())

##### 10. Co-occurrence analysis of coding categories
code_columns = [col for col in df.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\nCo-occurrence Counts Among Coding Categories:")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (df[col_a] == 1) & (df[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")

print("Inclusive language audit analysis completed.")
