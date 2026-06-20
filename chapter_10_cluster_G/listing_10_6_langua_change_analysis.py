import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load longitudinal dataset (output of Listing 10.5)
df = pd.read_csv(
    "out_G_langua_change_processed.csv",
    encoding='utf-8')
df = df.sort_values(
    ['Speaker_ID', 'Year']).reset_index(drop=True)
print(f"Observations: {len(df)}, "
      f"Speakers: {df['Speaker_ID'].nunique()}, "
      f"Years: {df['Year'].min():.0f}--"
      f"{df['Year'].max():.0f}")

##### 1. Aggregate annual variant frequencies
annual = df.groupby('Year').agg(
    N_Obs=('Outcome_Binary', 'count'),
    Freq_A=('Outcome_Binary', 'mean')
).reset_index()
annual['Freq_A'] = annual['Freq_A'].round(4)
print("\nAnnual variant frequencies:")
print(annual.to_string(index=False))

##### 2. Mixed-effects linear probability model
# Random intercept for Speaker_ID;
# fixed effect for Year_Centered and demographic
# predictors
pred_cols = ['Year_Centered']
if 'Education_Num' in df.columns:
    pred_cols.append('Education_Num')
formula = ("Outcome_Binary ~ "
           + " + ".join(pred_cols))
lmm_results = {}
try:
    valid_df = df.dropna(
        subset=pred_cols + ['Outcome_Binary'])
    lmm = smf.mixedlm(
        formula, data=valid_df,
        groups=valid_df['Speaker_ID']
    ).fit(reml=False)
    print(f"\n--- Mixed-Effects Model ---")
    print(lmm.summary())
    lmm_results = {
        'Year_C_Coef': round(
            lmm.params.get(
                'Year_Centered', np.nan), 6),
        'Year_C_pval': round(
            lmm.pvalues.get(
                'Year_Centered', np.nan), 6),
        'RE_Variance': round(
            float(lmm.cov_re.iloc[0, 0]), 6),
        'AIC': round(lmm.aic, 4),
        'N_Obs': int(lmm.nobs),
    }
    print(f"\nYear coefficient: "
          f"{lmm_results['Year_C_Coef']:.6f} "
          f"(p = {lmm_results['Year_C_pval']:.4f})")
except Exception as e:
    print(f"Mixed-effects model failed: {e}")

##### 3. Piecewise regression for change detection
# Split the time series at each possible breakpoint;
# choose the split that minimises total residual
# variance.
years_sorted = sorted(df['Year'].unique())
best_bp = None
if len(years_sorted) >= 6:
    print("\n--- Change Detection "
          "(Piecewise Regression) ---")
    best_rss = np.inf
    candidate_bps = years_sorted[2:-2]
    for bp in candidate_bps:
        seg1 = annual[annual['Year'] <= bp]
        seg2 = annual[annual['Year'] > bp]
        if len(seg1) < 2 or len(seg2) < 2:
            continue
        def seg_rss(seg):
            x = seg['Year'].values
            y = seg['Freq_A'].values
            if x.std() == 0:
                return ((y - y.mean()) ** 2).sum()
            r, _ = pearsonr(x, y)
            slope = r * y.std() / x.std()
            intercept = (y.mean()
                         - slope * x.mean())
            return (
                (y - (intercept + slope * x))
                ** 2).sum()
        rss = seg_rss(seg1) + seg_rss(seg2)
        if rss < best_rss:
            best_rss = rss
            best_bp  = bp
    print(f"Best breakpoint: {best_bp}")
    for label, seg in [
            ('Before',
             annual[annual['Year'] <= best_bp]),
            ('After',
             annual[annual['Year'] > best_bp])]:
        if len(seg) >= 2 and seg['Year'].std() > 0:
            r, p = pearsonr(
                seg['Year'], seg['Freq_A'])
            slope = (r * seg['Freq_A'].std()
                     / seg['Year'].std())
            print(f"  {label} {best_bp}: "
                  f"slope = {slope:.6f} per year, "
                  f"r = {r:.3f}, p = {p:.4f}")
else:
    print("Insufficient time points for change "
          "detection.")

##### 4. Prospective validation
# Train on first 70% of years; test on remaining 30%
split_year = np.percentile(years_sorted, 70)
train_df = df[df['Year'] <= split_year]
test_df  = df[df['Year'] > split_year]
print(f"\n--- Prospective Validation ---")
print(f"Train years: up to {split_year:.0f} "
      f"(n = {len(train_df)})")
print(f"Test years:  after {split_year:.0f} "
      f"(n = {len(test_df)})")
if len(train_df) >= 10 and len(test_df) >= 5:
    try:
        import statsmodels.api as sm
        X_tr = sm.add_constant(
            train_df[['Year_Centered']].fillna(0))
        y_tr = train_df['Outcome_Binary'].values
        ols_tr = sm.OLS(y_tr, X_tr).fit()
        X_te = sm.add_constant(
            test_df[['Year_Centered']].fillna(0))
        y_te = test_df['Outcome_Binary'].values
        y_pred = ols_tr.predict(X_te)
        rmse = np.sqrt(
            ((y_te - y_pred) ** 2).mean())
        mae = np.abs(y_te - y_pred).mean()
        print(f"Holdout RMSE: {rmse:.4f}, "
              f"MAE: {mae:.4f}")
    except Exception as e:
        print(f"Prospective validation failed: {e}")
else:
    print("Warning: insufficient data for prospective "
          "validation (need >= 10 train, >= 5 test).")

##### 5. Export results
annual.to_csv(
    "out_G_langua_change_annual_freq.csv",
    index=False)
if lmm_results:
    pd.DataFrame([lmm_results]).to_csv(
        "out_G_langua_change_lmm_results.csv",
        index=False)
if best_bp is not None:
    pd.DataFrame([{
        'Breakpoint_Year': best_bp}]).to_csv(
        "out_G_langua_change_changepoint.csv",
        index=False)
print("\nLanguage change longitudinal tracking "
      "analysis completed.")
print("Output files:")
print("  out_G_langua_change_annual_freq.csv")
print("  out_G_langua_change_lmm_results.csv")
print("  out_G_langua_change_changepoint.csv")
