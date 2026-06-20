import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Load node and edge data (output of Listing 10.17)
nodes = pd.read_csv(
    "out_G_langua_practi_nodes_processed.csv",
    encoding='utf-8')
edges = pd.read_csv(
    "out_G_langua_practi_edges_processed.csv",
    encoding='utf-8')
print(f"Speakers: {len(nodes)}, "
      f"Ties: {len(edges)}")

##### 1. Compute network metrics
speaker_ids = nodes['Speaker_ID'].tolist()

# Adjacency list
adj = {s: set() for s in speaker_ids}
for _, row in edges.iterrows():
    if (row['Speaker_A'] in adj
            and row['Speaker_B'] in adj):
        adj[row['Speaker_A']].add(
            row['Speaker_B'])
        adj[row['Speaker_B']].add(
            row['Speaker_A'])

# Degree centrality
degree = {s: len(adj[s]) for s in speaker_ids}
degree_norm = {
    s: degree[s] / max(1, len(speaker_ids) - 1)
    for s in speaker_ids}

# Clustering coefficient — indentation fixed
def clustering_coef(node, adj):
    neighbors = list(adj[node])
    k = len(neighbors)
    if k < 2:
        return 0.0
    links = sum(
        1 for i in range(k)
        for j in range(i + 1, k)
        if neighbors[j] in adj[neighbors[i]])
    return 2 * links / (k * (k - 1))

clustering = {
    s: clustering_coef(s, adj)
    for s in speaker_ids}

# Betweenness centrality (approximate via BFS)
def bfs_betweenness(adj, nodes):
    between = {n: 0.0 for n in nodes}
    for s in nodes:
        visited  = {s}
        queue    = [s]
        pred     = {n: [] for n in nodes}
        sigma    = {n: 0 for n in nodes}
        dist     = {n: -1 for n in nodes}
        sigma[s] = 1
        dist[s]  = 0
        stack    = []
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w)
                    visited.add(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {n: 0.0 for n in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (
                        sigma[v] / sigma[w]
                        * (1 + delta[w]))
            if w != s:
                between[w] += delta[w]
    n = len(nodes)
    norm = (n - 1) * (n - 2) if n > 2 else 1
    return {k: v / norm
            for k, v in between.items()}

print("Computing betweenness centrality...")
betweenness = bfs_betweenness(adj, speaker_ids)

# Merge metrics onto node file
nodes['Degree_Norm'] = nodes['Speaker_ID'].map(
    degree_norm)
nodes['Clustering']  = nodes['Speaker_ID'].map(
    clustering)
nodes['Betweenness'] = nodes['Speaker_ID'].map(
    betweenness)
nodes['Degree_Raw']  = nodes['Speaker_ID'].map(
    degree)
print(f"Mean degree: "
      f"{np.mean(list(degree.values())):.2f}, "
      f"Mean clustering: "
      f"{np.mean(list(clustering.values())):.3f}")

##### 2. Assemble feature columns
network_features = [
    'Degree_Norm', 'Clustering', 'Betweenness']
demo_features = [
    c for c in ['Age_Centered', 'Education_Num']
    if c in nodes.columns]
dummy_features = [
    c for c in nodes.columns
    if c.startswith('Gender_')]
all_features = (
    network_features + demo_features
    + dummy_features)

##### 3. Fit logistic regression with network
# predictors
model_data = nodes[
    all_features + ['Outcome_Binary']].dropna()
X = model_data[all_features].values
y = model_data['Outcome_Binary'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=(y if y.sum() > 3 else None))
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
lr = LogisticRegression(
    max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)
y_pred = lr.predict(X_te_s)
y_prob = lr.predict_proba(X_te_s)[:, 1]
acc = accuracy_score(y_te, y_pred)
try:
    auc = roc_auc_score(y_te, y_prob)
except ValueError:
    auc = np.nan
print(f"\nHoldout accuracy: {acc:.4f}, "
      f"AUC: {auc:.4f}")
coef_df = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': lr.coef_[0].round(4),
    'Odds_Ratio': np.exp(
        lr.coef_[0]).round(4),
}).sort_values('Coefficient', ascending=False)
print("\n--- Network + Demographic Effects ---")
print(coef_df.to_string(index=False))

##### 4. Network diffusion simulation
# Initialise agents from observed variant
# frequencies; at each step each agent adopts
# a random neighbor's variant.
N_STEPS = 20
N_SIMS  = 50
rng = np.random.default_rng(seed=42)
init_freq = nodes['Outcome_Binary'].mean()
diffusion_records = []
for sim in range(N_SIMS):
    states = {
        s: int(rng.random() < init_freq)
        for s in speaker_ids}
    step_freqs = [
        np.mean(list(states.values()))]
    for _ in range(N_STEPS):
        new_states = states.copy()
        for node in speaker_ids:
            neighbors = list(adj[node])
            if not neighbors:
                continue
            neighbor = rng.choice(neighbors)
            new_states[node] = (
                states[neighbor])
        states = new_states
        step_freqs.append(
            np.mean(list(states.values())))
    for step, freq in enumerate(step_freqs):
        diffusion_records.append({
            'Simulation': sim,
            'Step': step,
            'Freq_A': round(freq, 4),
        })
diffusion_df = pd.DataFrame(diffusion_records)
summary = diffusion_df.groupby('Step').agg(
    Mean_Freq=('Freq_A', 'mean'),
    SD_Freq=('Freq_A', 'std')
).reset_index()
print(f"\nDiffusion trajectory (first 5 steps):")
print(summary.head(6).to_string(index=False))

##### 5. Export results
nodes[[
    'Speaker_ID', 'Degree_Raw',
    'Degree_Norm', 'Clustering',
    'Betweenness', 'Outcome_Binary'
]].to_csv(
    "out_G_langua_practi_metrics.csv",
    index=False)
coef_df.to_csv(
    "out_G_langua_practi_coefficients.csv",
    index=False)
summary.to_csv(
    "out_G_langua_practi_diffusion.csv",
    index=False)
pd.DataFrame([{
    'Holdout_Accuracy': round(acc, 4),
    'Holdout_AUC': (round(auc, 4)
                    if not np.isnan(auc)
                    else None),
}]).to_csv(
    "out_G_langua_practi_model_fit.csv",
    index=False)
print("\nLanguage practice network mapping "
      "analysis completed.")
print("Output files:")
print("  out_G_langua_practi_metrics.csv")
print("  out_G_langua_practi_coefficients.csv")
print("  out_G_langua_practi_diffusion.csv")
print("  out_G_langua_practi_model_fit.csv")
