import pandas as pd
import numpy as np
from itertools import combinations

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster D block)
df = pd.read_csv("out_D_geolin_mappin_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Warn on duplicate Speaker_ID + Feature records
# The universal block deduplicates identical rows; this
# check catches logically duplicate observations where
# the same speaker has two conflicting variants for the
# same feature.
dupes = df.duplicated(subset=['Speaker_ID', 'Feature'])
if dupes.any():
    n = dupes.sum()
    print(f"Warning: {n} duplicate Speaker_ID + Feature "
          f"records detected. Keeping first occurrence.")
    df = df.drop_duplicates(
        subset=['Speaker_ID', 'Feature'], keep='first')

##### 2. Build site-level coordinate table
# Each site should have consistent coordinates across
# speakers; mean is taken in case of minor rounding.
site_coords = (df.groupby('Site_ID')
               [['Latitude', 'Longitude']]
               .mean().reset_index())
site_coords.columns = ['Site_ID', 'Site_Lat', 'Site_Lon']

# Check for sites with large coordinate variance
coord_var = (df.groupby('Site_ID')
             [['Latitude', 'Longitude']]
             .std().fillna(0))
suspicious = coord_var[
    (coord_var['Latitude'] > 0.01) |
    (coord_var['Longitude'] > 0.01)]
if not suspicious.empty:
    print(f"Warning: {len(suspicious)} sites with "
          f"inconsistent coordinates across speakers: "
          f"{suspicious.index.tolist()}")

##### 3. Compute per-site variant frequencies per feature
freq = (df.groupby(['Site_ID', 'Feature', 'Variant'])
        .size().reset_index(name='Count'))
totals = (freq.groupby(['Site_ID', 'Feature'])['Count']
          .sum().reset_index(name='Total'))
freq = freq.merge(totals, on=['Site_ID', 'Feature'])
freq['Frequency'] = freq['Count'] / freq['Total']
speaker_counts = (df.groupby(['Site_ID', 'Feature'])
                  ['Speaker_ID'].nunique()
                  .reset_index(name='N_Speakers'))
freq = freq.merge(speaker_counts,
                  on=['Site_ID', 'Feature'])

if 'Feature_Category' in df.columns:
    feat_cat = (df[['Feature', 'Feature_Category']]
                .drop_duplicates()
                .groupby('Feature')['Feature_Category']
                .first().reset_index())
    freq = freq.merge(feat_cat, on='Feature', how='left')

freq = freq.merge(site_coords, on='Site_ID', how='left')
print(f"Frequency dataset: {len(freq)} rows across "
      f"{freq['Site_ID'].nunique()} sites and "
      f"{freq['Feature'].nunique()} features.")

##### 4. Identify dominant variant per site per feature
dominant = (freq.loc[
    freq.groupby(['Site_ID', 'Feature'])
    ['Frequency'].idxmax()]
    [['Site_ID', 'Feature', 'Variant', 'Frequency',
      'N_Speakers', 'Site_Lat', 'Site_Lon']]
    .rename(columns={
        'Variant': 'Dominant_Variant',
        'Frequency': 'Dominant_Frequency'})
    .reset_index(drop=True))

if 'Feature_Category' in freq.columns:
    feat_cat = (freq[['Feature', 'Feature_Category']]
                .drop_duplicates())
    dominant = dominant.merge(
        feat_cat, on='Feature', how='left')

print(f"Dominant variant table: {len(dominant)} "
      f"site-feature pairs.")

##### 5. Compute pairwise inter-site geographic distances
# Haversine formula for great-circle distance in km.
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

sites = site_coords.copy()
distance_records = []
for s1, s2 in combinations(range(len(sites)), 2):
    row1 = sites.iloc[s1]
    row2 = sites.iloc[s2]
    dist = haversine(
        row1['Site_Lat'], row1['Site_Lon'],
        row2['Site_Lat'], row2['Site_Lon'])
    distance_records.append({
        'Site_A': row1['Site_ID'],
        'Site_B': row2['Site_ID'],
        'Distance_km': round(dist, 4)})
distances = pd.DataFrame(distance_records)
print(f"Inter-site distance table: {len(distances)} "
      f"site pairs.")

##### 6. Save outputs
freq.to_csv(
    "out_D_geolin_mappin_frequencies.csv", index=False)
dominant.to_csv(
    "out_D_geolin_mappin_dominant.csv", index=False)
site_coords.to_csv(
    "out_D_geolin_mappin_sites.csv", index=False)
distances.to_csv(
    "out_D_geolin_mappin_distances.csv", index=False)
print("Geolinguistic mapping data processing completed.")
print("Output files:")
print("  out_D_geolin_mappin_frequencies.csv  "
      "-- long-format variant frequencies")
print("  out_D_geolin_mappin_dominant.csv     "
      "-- dominant variant per site/feature")
print("  out_D_geolin_mappin_sites.csv        "
      "-- site coordinates")
print("  out_D_geolin_mappin_distances.csv    "
      "-- pairwise inter-site distances")
