import pandas as pd
import numpy as np
import re
from collections import Counter

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster E block)
df = pd.read_csv("out_E_social_media_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Parse and normalize timestamps
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], errors='coerce')
    invalid_ts = df['Timestamp'].isna()
    if invalid_ts.any():
        print(f"Warning: {invalid_ts.sum()} records "
              f"with unparseable Timestamp values.")
    df['Year']  = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Date']  = df['Timestamp'].dt.date
    print(f"Timestamp range: "
          f"{df['Timestamp'].min()} to "
          f"{df['Timestamp'].max()}")

##### 2. Normalize platform field
if 'Platform' in df.columns:
    df['Platform'] = (df['Platform'].str.strip()
                      .str.lower())

##### 3. Flag bot/automated content
# Flags posts matching common bot patterns: URL-only
# posts, excessive hashtag density, all-caps short posts.
def is_likely_bot(text):
    tokens = str(text).split()
    if len(tokens) == 0:
        return False
    url_ratio = sum(
        1 for t in tokens
        if t.startswith('http')) / len(tokens)
    hashtag_ratio = sum(
        1 for t in tokens
        if t.startswith('#')) / len(tokens)
    all_caps = (text == text.upper()
                and len(text) > 10
                and text.isalpha())
    return (url_ratio > 0.8
            or hashtag_ratio > 0.7
            or all_caps)

df['Bot_Flag'] = (df['Text']
                  .apply(is_likely_bot).astype(int))
n_bot = df['Bot_Flag'].sum()
if n_bot > 0:
    pct = n_bot / len(df) * 100
    print(f"Bot/automated content flagged: {n_bot} "
          f"posts ({pct:.1f}%). Retained; filter "
          f"before analysis.")

##### 4. Extract platform-specific tokens
# Extracted before normalization so hashtags, mentions,
# and URLs remain available as sociolinguistic data.
def extract_hashtags(text):
    return re.findall(r"#\w+", str(text))

def extract_mentions(text):
    return re.findall(r"@\w+", str(text))

def extract_urls(text):
    return re.findall(r"https?://\S+", str(text))

df['Hashtags'] = df['Text'].apply(extract_hashtags)
df['Mentions'] = df['Text'].apply(extract_mentions)
df['URLs']     = df['Text'].apply(extract_urls)
df['N_Hashtags'] = df['Hashtags'].apply(len)
df['N_Mentions'] = df['Mentions'].apply(len)
df['N_URLs']     = df['URLs'].apply(len)
# Serialize list columns to pipe-separated strings
for col in ['Hashtags', 'Mentions', 'URLs']:
    df[col] = df[col].apply(
        lambda x: '|'.join(x) if x else '')

##### 5. Normalize text for lexical analysis
# Produces a cleaned version for token-frequency
# analysis. Preserves the original Text column.
def normalize_text(text):
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Text_Normalized'] = (df['Text']
                         .apply(normalize_text))
df['Text_Lower'] = df['Text_Normalized'].str.lower()

##### 6. Compute basic post statistics
df['Token_Count'] = df['Text_Normalized'].apply(
    lambda x: len(str(x).split()))
df['Char_Count'] = df['Text_Normalized'].str.len()
df = df[df['Token_Count'] > 0]
print(f"Posts retained after normalization: {len(df)}")

##### 7. Token-level language identification heuristic
# Character n-gram heuristic for language profile
# detection. Adapt regex patterns to the language pair
# under study. Example below targets Filipino/English.
TAGALOG_PATTERNS = re.compile(
    r"\b(ang|ng|sa|na|mga|ay|ko|mo|siya|ito|iyan|"
    r"nang|pa|din|rin|hindi|pero|kasi|talaga|sobra|"
    r"grabe|naman|lang|po|opo|mag|nag|pag|um|in|an)"
    r"\b", re.IGNORECASE)
ENGLISH_PATTERNS = re.compile(
    r"\b(the|a|an|is|are|was|were|it|this|that|and|"
    r"but|or|not|have|has|had|will|would|can|could|"
    r"should|be|been|being|with|for|from|they|we|you|"
    r"he|she)\b", re.IGNORECASE)

def detect_languages(text):
    tl = len(TAGALOG_PATTERNS.findall(str(text)))
    en = len(ENGLISH_PATTERNS.findall(str(text)))
    total = tl + en
    if total == 0:
        return "undetermined"
    if tl / total > 0.7:
        return "L1_dominant"
    if en / total > 0.7:
        return "L2_dominant"
    return "mixed"

df['Language_Profile'] = (df['Text_Lower']
                          .apply(detect_languages))
print(f"\nLanguage profile distribution:")
print(df['Language_Profile'].value_counts())

##### 8. Flag code-switched posts
df['Code_Switch_Flag'] = (
    df['Language_Profile'] == "mixed").astype(int)
n_cs = df['Code_Switch_Flag'].sum()
print(f"Code-switched posts detected: {n_cs} "
      f"({n_cs / len(df) * 100:.1f}%)")

##### 9. Per-user aggregation for diffusion analysis
agg_dict = {
    'N_Posts':         ('Post_ID', 'count'),
    'Total_Tokens':    ('Token_Count', 'sum'),
    'N_Hashtags_Total':('N_Hashtags', 'sum'),
    'N_CS_Posts':      ('Code_Switch_Flag', 'sum'),
}
if 'Timestamp' in df.columns:
    agg_dict['First_Post'] = ('Timestamp', 'min')
    agg_dict['Last_Post']  = ('Timestamp', 'max')
user_stats = (df.groupby('User_ID')
              .agg(**agg_dict).reset_index())
user_stats['CS_Rate'] = (
    user_stats['N_CS_Posts']
    / user_stats['N_Posts']).round(4)
if ('Location' in df.columns):
    user_location = (
        df[['User_ID', 'Location']]
        .drop_duplicates('User_ID')
        .dropna(subset=['Location']))
    user_stats = user_stats.merge(
        user_location, on='User_ID', how='left')

##### 10. Save outputs
df_out = df.drop(columns=['Text_Lower'],
                 errors='ignore')
df_out.to_csv(
    "out_E_social_media_processed.csv", index=False)
user_stats.to_csv(
    "out_E_social_media_user_stats.csv", index=False)
print("\nSocial media language analysis data processing "
      "completed.")
print("Output files:")
print("  out_E_social_media_processed.csv   "
      "-- cleaned corpus with language profiles, "
      "code-switch flags, extracted tokens")
print("  out_E_social_media_user_stats.csv  "
      "-- per-user aggregates for diffusion analysis")
