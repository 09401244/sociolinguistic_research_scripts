import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Load processed datasets (output of Listing 7.3)
nodes = pd.read_csv(
    "out_D_social_networ_nodes_processed.csv",
    encoding='utf-8')
edges = pd.read_csv(
    "out_D_social_networ_edges_processed.csv",
    encoding='utf-8')
feature_cols = [
    col for col in nodes.columns
    if col.startswith('Feature_')
    and col.endswith('_Variant')]

##### 1. Global network properties
n_nodes = len(nodes)
n_edges = (edges.drop_duplicates(
    subset=['Node_A', 'Node_B']).shape[0])
max_possible = n_nodes * (n_nodes - 1) / 2
density = (n_edges / max_possible
           if max_possible > 0 else 0)
print("=== Global Network Properties ===")
print(f"Actors (nodes):           {n_nodes}")
print(f"Unique ties (edges):      {n_edges}")
print(f"Network density:          {density:.4f}")
print(f"Mean degree:              "
      f"{nodes['Degree'].mean():.2f}")
print(f"Mean multiplexity:        "
      f"{nodes['Mean_Multiplexity'].mean():.2f}")
print(f"Mean network integration: "
      f"{nodes['Network_Integration'].mean():.2f}")
if 'Tie_Strength' in edges.columns:
    strong = (edges['Tie_Strength'] == 'strong').sum()
    weak   = (edges['Tie_Strength'] == 'weak').sum()
    print(f"Strong ties:              "
          f"{strong} ({100*strong/len(edges):.1f}%)")
    print(f"Weak ties:                "
          f"{weak} ({100*weak/len(edges):.1f}%)")

##### 2. Tie type distribution
print("\n=== Tie Type Distribution ===")
tie_counts = edges['Tie_Type'].value_counts()
print(tie_counts)
tie_counts.plot(kind='bar')
plt.title("Distribution of Tie Types")
plt.xlabel("Tie Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

##### 3. Network integration score distribution
print("\n=== Network Integration Score Distribution ===")
print(nodes['Network_Integration'].describe())
nodes['Network_Integration'].hist(bins=15)
plt.title("Distribution of Network Integration Scores")
plt.xlabel("Network Integration Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# Integration by social group attributes
for attr in ['Gender', 'Occupation', 'Neighborhood']:
    if attr in nodes.columns:
        print(f"\nMean Network Integration by {attr}:")
        print(nodes.groupby(attr)
              ['Network_Integration'].mean()
              .sort_values(ascending=False).round(3))

##### 4. Network structure and linguistic variant use
print("\n=== Network Structure and Linguistic "
      "Variant Use ===")
for feature_col in feature_cols:
    feature_name = (feature_col
                    .replace('Feature_', '')
                    .replace('_Variant', ''))
    print(f"\n--- Feature: {feature_name} ---")
    sub = nodes[nodes[feature_col].notna()].copy()
    variants = sub[feature_col].unique()
    if len(variants) < 2:
        print(f"  Only one variant present. Skipping.")
        continue
    print("  Variant frequencies:")
    print(sub[feature_col].value_counts())
    print("\n  Mean Network Integration by Variant:")
    print(sub.groupby(feature_col)
          ['Network_Integration'].mean().round(3))
    print("\n  Mean Degree by Variant:")
    print(sub.groupby(feature_col)
          ['Degree'].mean().round(2))
    print("\n  Mean Multiplexity by Variant:")
    print(sub.groupby(feature_col)
          ['Mean_Multiplexity'].mean().round(3))
    # Kruskal-Wallis test
    groups = [
        sub.loc[sub[feature_col] == v,
                'Network_Integration']
        .dropna().values
        for v in variants]
    if all(len(g) > 1 for g in groups):
        stat, p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis test "
              f"(Network Integration ~ Variant): "
              f"H={stat:.3f}, p={p:.3f}")
    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    variant_data = [
        sub.loc[sub[feature_col] == v,
                'Network_Integration']
        .dropna().values
        for v in variants]
    ax.boxplot(variant_data, labels=variants)
    ax.set_title(f"Network Integration by Variant: "
                 f"{feature_name}")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Network Integration Score")
    plt.tight_layout()
    plt.show()

##### 5. Strong vs. weak tie effects on variant use
if ('Strong_Ties' in nodes.columns
        and 'Weak_Ties' in nodes.columns):
    print("\n=== Strong vs. Weak Tie Effects on "
          "Variant Use ===")
    for feature_col in feature_cols:
        feature_name = (feature_col
                        .replace('Feature_', '')
                        .replace('_Variant', ''))
        sub = nodes[nodes[feature_col].notna()].copy()
        if sub[feature_col].nunique() < 2:
            continue
        print(f"\n--- Feature: {feature_name} ---")
        print("  Mean Strong Ties by Variant:")
        print(sub.groupby(feature_col)
              ['Strong_Ties'].mean().round(2))
        print("  Mean Weak Ties by Variant:")
        print(sub.groupby(feature_col)
              ['Weak_Ties'].mean().round(2))

##### 6. Multiplexity and variant conservatism
print("\n=== Multiplexity and Variant Use ===")
for feature_col in feature_cols:
    feature_name = (feature_col
                    .replace('Feature_', '')
                    .replace('_Variant', ''))
    sub = nodes[nodes[feature_col].notna()].copy()
    if sub[feature_col].nunique() < 2:
        continue
    print(f"\n  Feature: {feature_name}")
    print(sub.groupby(feature_col)
          ['Mean_Multiplexity']
          .agg(['mean', 'median', 'std']).round(3))

##### 7. Variant homophily within ties
print("\n=== Variant Homophily Within Ties ===")
for feature_col in feature_cols:
    feature_name = (feature_col
                    .replace('Feature_', '')
                    .replace('_Variant', ''))
    node_variant = (nodes.set_index('Node_ID')
                    [feature_col])
    edges_filt = edges[
        edges['Node_A'].isin(node_variant.index)
        & edges['Node_B'].isin(
            node_variant.index)].copy()
    edges_filt['Variant_A'] = (edges_filt['Node_A']
                                .map(node_variant))
    edges_filt['Variant_B'] = (edges_filt['Node_B']
                                .map(node_variant))
    edges_filt = edges_filt.dropna(
        subset=['Variant_A', 'Variant_B'])
    if len(edges_filt) == 0:
        continue
    edges_filt['Same_Variant'] = (
        edges_filt['Variant_A']
        == edges_filt['Variant_B'])
    homophily = edges_filt['Same_Variant'].mean()
    print(f"\n  Feature: {feature_name}")
    print(f"  Proportion of ties linking same-variant "
          f"actors: {homophily:.3f}")
    if 'Tie_Strength' in edges_filt.columns:
        print("  By tie strength:")
        print(edges_filt.groupby('Tie_Strength')
              ['Same_Variant'].mean().round(3))

##### 8. Co-occurrence analysis of coding categories
code_columns = [col for col in nodes.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\n=== Co-occurrence Counts Among "
          "Coding Categories ===")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (nodes[col_a] == 1)
            & (nodes[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")

print("\nSocial network analysis completed.")
