import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load model-ready dataset (output of Listing 8.3)
df = pd.read_csv(
    "out_E_causal_analys_processed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")
print(f"Outcome distribution:\n"
      f"{df['Outcome_Binary'].value_counts()}")

# Identify predictor columns automatically
numeric_preds = [
    c for c in ['Age_Centered', 'Education_Numeric']
    if c in df.columns]
dummy_preds = [
    c for c in df.columns
    if c.startswith(('Gender_', 'Register_',
                     'LingCtx_'))]
all_preds = numeric_preds + dummy_preds
print(f"Predictors: {all_preds}")
print(f"Grouping variable (random effects): Speaker_ID")

##### 1. Baseline logistic regression (fixed effects)
X_base = sm.add_constant(df[all_preds].fillna(0))
y = df['Outcome_Binary']
logit_model = sm.Logit(y, X_base).fit(disp=0)
print("\n--- Baseline Logistic Regression ---")
print(logit_model.summary2())

# Coefficients, standard errors, odds ratios, 95% CIs
coef_df = pd.DataFrame({
    'Predictor': logit_model.params.index,
    'Coefficient': (logit_model.params.values
                    .round(4)),
    'Std_Error': logit_model.bse.values.round(4),
    'z_value': logit_model.tvalues.values.round(4),
    'p_value': logit_model.pvalues.values.round(6),
    'Odds_Ratio': np.exp(
        logit_model.params.values).round(4),
    'CI_Lower_OR': np.exp(
        logit_model.conf_int()[0].values).round(4),
    'CI_Upper_OR': np.exp(
        logit_model.conf_int()[1].values).round(4),
    'Significant_p05': (
        logit_model.pvalues.values < 0.05),
})
print("\n--- Odds Ratios and 95% CIs ---")
print(coef_df.to_string(index=False))

##### 2. Model fit statistics for baseline model
fit_stats_base = {
    'Model': 'Baseline_Logistic',
    'N_Observations': int(logit_model.nobs),
    'Log_Likelihood': round(logit_model.llf, 4),
    'AIC': round(logit_model.aic, 4),
    'BIC': round(logit_model.bic, 4),
    'Pseudo_R2_McFadden': round(
        logit_model.prsquared, 4),
}
print(f"\nBaseline model AIC: {fit_stats_base['AIC']}, "
      f"McFadden R2: "
      f"{fit_stats_base['Pseudo_R2_McFadden']}")

##### 3. Mixed-effects model accounting for
# speaker-level clustering.
# statsmodels MixedLM operates on a continuous outcome.
# We use a linear probability model (LPM) approximation
# here as a diagnostic for random-effects variance;
# for production analyses, consider pymer4.
fit_stats_mixed = {}
if df['Speaker_ID'].nunique() >= 5:
    formula_parts = (
        ' + '.join(all_preds) if all_preds else '1')
    formula = f"Outcome_Binary ~ {formula_parts}"
    try:
        lpm_mixed = smf.mixedlm(
            formula, data=df.fillna(0),
            groups=df['Speaker_ID']).fit(reml=False)
        print("\n--- Linear Mixed Model "
              "(LPM approximation) ---")
        print(lpm_mixed.summary())
        fit_stats_mixed = {
            'Model': 'Mixed_LPM',
            'N_Observations': int(lpm_mixed.nobs),
            'Log_Likelihood': round(lpm_mixed.llf, 4),
            'AIC': round(lpm_mixed.aic, 4),
            'BIC': round(lpm_mixed.bic, 4),
            'Random_Effects_Variance': round(
                float(lpm_mixed.cov_re.iloc[0, 0]),
                6),
        }
        print(f"Speaker-level random-effects variance: "
              f"{fit_stats_mixed['Random_Effects_Variance']}")
        print(f"Mixed model AIC: "
              f"{fit_stats_mixed['AIC']}")
    except Exception as e:
        print(f"Mixed model failed: {e}")
else:
    print("Insufficient speakers for mixed-effects "
          "modeling (minimum 5 required).")

##### 4. Likelihood ratio test: baseline vs. null model
X_null = sm.add_constant(
    pd.Series(np.ones(len(y)), name='const'))
logit_null = sm.Logit(y, X_null).fit(disp=0)
lr_stat = -2 * (logit_null.llf - logit_model.llf)
lr_df = logit_model.df_model
from scipy.stats import chi2 as chi2_dist
lr_p = chi2_dist.sf(lr_stat, lr_df)
print(f"\n--- Likelihood Ratio Test "
      f"(full vs. null) ---")
print(f"LR statistic: {lr_stat:.4f}, "
      f"df: {lr_df}, p-value: {lr_p:.6f}")

##### 5. Predicted probabilities and residual
# diagnostics
df['Predicted_Prob'] = logit_model.predict(X_base)
df['Predicted_Class'] = (
    df['Predicted_Prob'] >= 0.5).astype(int)
accuracy = (
    df['Predicted_Class']
    == df['Outcome_Binary']).mean()
print(f"\nClassification accuracy "
      f"(threshold = 0.5): {accuracy:.4f}")
try:
    from sklearn.metrics import (
        confusion_matrix, classification_report)
    cm = confusion_matrix(
        df['Outcome_Binary'], df['Predicted_Class'])
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(
        df['Outcome_Binary'],
        df['Predicted_Class']))
except ImportError:
    print("sklearn not available; "
          "skipping confusion matrix.")

##### 6. Export results
# Use Text_ID as the observation identifier
# (Token_ID not produced by preprocessing script)
id_col = ('Text_ID' if 'Text_ID' in df.columns
          else df.columns[0])
coef_df.to_csv(
    "out_E_causal_analys_coefficients.csv",
    index=False)
fit_df = pd.DataFrame([fit_stats_base])
if fit_stats_mixed:
    fit_df = pd.concat(
        [fit_df, pd.DataFrame([fit_stats_mixed])],
        ignore_index=True)
fit_df.to_csv(
    "out_E_causal_analys_model_fit.csv", index=False)
df[[id_col, 'Speaker_ID', 'Outcome_Binary',
    'Predicted_Prob', 'Predicted_Class']].to_csv(
    "out_E_causal_analys_predictions.csv",
    index=False)
pd.DataFrame([{
    'LR_Statistic': round(lr_stat, 4),
    'LR_df': lr_df,
    'LR_p_value': round(lr_p, 6),
    'Significant_p05': lr_p < 0.05}]).to_csv(
    "out_E_causal_analys_lrtest.csv", index=False)
print("\nCausal analysis of language variation data "
      "analysis completed.")
print("Output files:")
print("  out_E_causal_analys_coefficients.csv")
print("  out_E_causal_analys_model_fit.csv")
print("  out_E_causal_analys_predictions.csv")
print("  out_E_causal_analys_lrtest.csv")
