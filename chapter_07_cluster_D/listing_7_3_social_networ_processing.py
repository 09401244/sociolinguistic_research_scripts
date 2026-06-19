import pandas as pd
import numpy as np

# Load preprocessed node dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster D block)
nodes = pd.read_csv(
    "out_D_social_networ_preprocessed.csv",
    encoding='utf-8')
print(f"Node file: {len(nodes)} actors loaded.")

# Load raw edge file directly (second input file;
# not produced by the preprocessing script —
# prepare this file separately before running)
edges = pd.read_csv("sna_edges_raw.csv",
                    encoding='utf-8')
edges = edges.drop_duplicates()
edges = edges.dropna(subset=['Node_A', 'Node_B'])
edges['Node_A'] = (edges['Node_A'].str.strip()
                   .str.upper())
edges['Node_B'] = (edges['Node_B'].str.strip()
                   .str.upper())
edges['Tie_Type'] = (edges['Tie_Type'].str.strip()
                     .str.lower())
print(f"Edge file: {len(edges)} ties loaded.")

##### 1. Validate tie strength labels
if 'Tie_Strength' in edges.columns:
    edges['Tie_Strength'] = (edges['Tie_Strength']
                             .str.strip().str.lower())
    valid_strength = {'strong', 'weak'}
    invalid = ~edges['Tie_Strength'].isin(
        valid_strength)
    if invalid.any():
        print(f"Warning: {invalid.sum()} edges with "
              f"unrecognized Tie_Strength values.")

##### 2. Remove self-loops
self_loops = edges['Node_A'] == edges['Node_B']
if self_loops.any():
    print(f"Warning: {self_loops.sum()} self-loop "
          f"edges removed.")
    edges = edges[~self_loops]

##### 3. Canonicalize edge direction (undirected network)
edges[['Node_A', 'Node_B']] = pd.DataFrame(
    edges[['Node_A', 'Node_B']].apply(
        lambda r: sorted([r['Node_A'], r['Node_B']]),
        axis=1).tolist(),
    index=edges.index)

##### 4. Flag nodes in edge file absent from node file
edge_nodes = set(edges['Node_A']) | set(edges['Node_B'])
node_set = set(nodes['Node_ID'])
missing = edge_nodes - node_set
if missing:
    print(f"Warning: {len(missing)} node(s) appear in "
          f"edge file but not in node file: {missing}")
print(f"Edge file validated: {len(edges)} ties across "
      f"{edges['Tie_Type'].nunique()} tie type(s).")

##### 5. Compute per-node degree
degree_a = edges.groupby('Node_A')['Node_B'].nunique()
degree_b = edges.groupby('Node_B')['Node_A'].nunique()
degree = (degree_a.add(degree_b, fill_value=0)
          .reset_index())
degree.columns = ['Node_ID', 'Degree']
nodes = nodes.merge(degree, on='Node_ID', how='left')
nodes['Degree'] = nodes['Degree'].fillna(0).astype(int)

##### 6. Compute strong and weak tie counts per node
if 'Tie_Strength' in edges.columns:
    for strength in ['strong', 'weak']:
        sub = edges[edges['Tie_Strength'] == strength]
        deg_a = sub.groupby('Node_A')['Node_B'].nunique()
        deg_b = sub.groupby('Node_B')['Node_A'].nunique()
        col = f"{strength.capitalize()}_Ties"
        deg = (deg_a.add(deg_b, fill_value=0)
               .reset_index())
        deg.columns = ['Node_ID', col]
        nodes = nodes.merge(deg, on='Node_ID', how='left')
        nodes[col] = nodes[col].fillna(0).astype(int)

##### 7. Compute multiplexity per edge pair
multiplexity = (edges.groupby(['Node_A', 'Node_B'])
                ['Tie_Type'].nunique().reset_index()
                .rename(columns={
                    'Tie_Type': 'Multiplexity'}))
edges = edges.merge(
    multiplexity, on=['Node_A', 'Node_B'], how='left')
mpx_a = edges.groupby('Node_A')['Multiplexity'].mean()
mpx_b = edges.groupby('Node_B')['Multiplexity'].mean()
mpx = ((mpx_a.add(mpx_b, fill_value=0) / 2)
       .reset_index())
mpx.columns = ['Node_ID', 'Mean_Multiplexity']
nodes = nodes.merge(mpx, on='Node_ID', how='left')
nodes['Mean_Multiplexity'] = (nodes['Mean_Multiplexity']
                              .fillna(0))

##### 8. Compute network integration score
# Combines degree and mean multiplexity following
# the logic of Milroy's network strength scale.
# Both components are min-max normalized first.
def minmax(series):
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(
            np.zeros(len(series)), index=series.index)
    return (series - series.min()) / rng

nodes['Degree_Norm'] = minmax(nodes['Degree'])
nodes['Multiplexity_Norm'] = minmax(
    nodes['Mean_Multiplexity'])
nodes['Network_Integration'] = (
    (nodes['Degree_Norm']
     + nodes['Multiplexity_Norm']) / 2)

##### 9. Normalize linguistic feature variant columns
feature_cols = [
    col for col in nodes.columns
    if col.startswith('Feature_')
    and col.endswith('_Variant')]
for col in feature_cols:
    nodes[col] = nodes[col].str.strip()
print(f"Linguistic feature columns detected: "
      f"{feature_cols}")

##### 10. Save processed outputs
edges_deduped = edges.drop_duplicates(
    subset=['Node_A', 'Node_B', 'Tie_Type'])
nodes.to_csv(
    "out_D_social_networ_nodes_processed.csv",
    index=False)
edges_deduped.to_csv(
    "out_D_social_networ_edges_processed.csv",
    index=False)
print("Social network analysis data processing "
      "completed.")
print("Output files:")
print("  out_D_social_networ_nodes_processed.csv  "
      "-- node attributes + network metrics")
print("  out_D_social_networ_edges_processed.csv  "
      "-- validated edges + multiplexity")
