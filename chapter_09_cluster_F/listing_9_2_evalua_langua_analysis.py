import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load processed dataset (output of Listing 9.1)
df = pd.read_csv(
    "out_F_evalua_langua_processed.csv",
    encoding='utf-8')
print(f"Participants loaded: {len(df)}")
print(f"Group distribution:\n"
      f"{df['Group'].value_counts()}")

# Identify outcome measures present in the dataset
outcome_bases = [
    b for b in
    ['Attitude_Score', 'Language_Use_Score']
    if (f"{b}_pre" in df.columns
        and f"{b}_post" in df.columns)]
print(f"Outcome measures: {outcome_bases}")

# Separate treatment and control groups
treatment = df[df['Group'] == 'treatment'].copy()
control   = df[df['Group'] == 'control'].copy()
print(f"Treatment n = {len(treatment)}, "
      f"Control n = {len(control)}")

##### 1. Descriptive statistics by group and time point
print("\n--- Descriptive Statistics ---")
desc_records = []
for base in outcome_bases:
    for grp_name, grp_df in [
            ('treatment', treatment),
            ('control', control)]:
        for time in ['pre', 'post']:
            col = f"{base}_{time}"
            if col not in grp_df.columns:
                continue
            s = grp_df[col].describe()
            print(f"{base} | {grp_name} | {time}: "
                  f"mean={s['mean']:.3f}, "
                  f"sd={s['std']:.3f}, "
                  f"n={int(s['count'])}")
            desc_records.append({
                'Outcome': base,
                'Group': grp_name,
                'Time': time,
                'N': int(s['count']),
                'Mean': round(s['mean'], 4),
                'SD':   round(s['std'], 4),
                'Min':  round(s['min'], 4),
                'Max':  round(s['max'], 4),
            })

##### 2. Paired t-test: pre vs. post within
# treatment group
print("\n--- Paired T-Tests: Pre vs. Post "
      "(Treatment) ---")
paired_results = []
for base in outcome_bases:
    pre_col  = f"{base}_pre"
    post_col = f"{base}_post"
    valid = treatment[[pre_col, post_col]].dropna()
    if len(valid) < 3:
        print(f"{base}: insufficient data for "
              f"paired t-test.")
        continue
    t_stat, p_val = stats.ttest_rel(
        valid[pre_col], valid[post_col])
    diff = valid[post_col] - valid[pre_col]
    cohens_d = diff.mean() / diff.std(ddof=1)
    print(f"{base}: t({len(valid)-1}) = {t_stat:.3f}, "
          f"p = {p_val:.4f}, d = {cohens_d:.3f}")
    paired_results.append({
        'Outcome': base,
        'Test': 'Paired_t_Treatment',
        'N': len(valid),
        't_statistic': round(t_stat, 4),
        'df': len(valid) - 1,
        'p_value': round(p_val, 6),
        'Cohens_d': round(cohens_d, 4),
        'Significant_p05': p_val < 0.05,
    })

##### 3. Independent t-test: post scores,
# treatment vs. control
print("\n--- Independent T-Tests: Post Scores ---")
between_results = []
for base in outcome_bases:
    post_col = f"{base}_post"
    t_grp = treatment[post_col].dropna()
    c_grp = control[post_col].dropna()
    if len(t_grp) < 3 or len(c_grp) < 3:
        print(f"{base}: insufficient data for "
              f"independent t-test.")
        continue
    t_stat, p_val = stats.ttest_ind(t_grp, c_grp)
    pooled_sd = np.sqrt(
        ((len(t_grp)-1) * t_grp.std(ddof=1)**2
         + (len(c_grp)-1) * c_grp.std(ddof=1)**2)
        / (len(t_grp) + len(c_grp) - 2))
    cohens_d = (
        (t_grp.mean() - c_grp.mean()) / pooled_sd
        if pooled_sd > 0 else np.nan)
    print(f"{base}: t({len(t_grp)+len(c_grp)-2}) = "
          f"{t_stat:.3f}, p = {p_val:.4f}, "
          f"d = {cohens_d:.3f}")
    between_results.append({
        'Outcome': base,
        'Test': 'Independent_t_Post',
        'N_Treatment': len(t_grp),
        'N_Control': len(c_grp),
        'Mean_Treatment': round(t_grp.mean(), 4),
        'Mean_Control':   round(c_grp.mean(), 4),
        't_statistic': round(t_stat, 4),
        'df': len(t_grp) + len(c_grp) - 2,
        'p_value': round(p_val, 6),
        'Cohens_d': round(cohens_d, 4),
        'Significant_p05': p_val < 0.05,
    })

##### 4. Implementation fidelity and outcome change
print("\n--- Fidelity-Outcome Correlations "
      "(Treatment) ---")
fidelity_results = []
if 'Implementation_Fidelity' in treatment.columns:
    for base in outcome_bases:
        change_col = f"{base}_Change"
        if change_col not in treatment.columns:
            continue
        valid = treatment[[
            'Implementation_Fidelity',
            change_col]].dropna()
        if len(valid) < 5:
            print(f"{base}: insufficient data for "
                  f"fidelity correlation.")
            continue
        r, p_val = stats.pearsonr(
            valid['Implementation_Fidelity'],
            valid[change_col])
        print(f"{base}: r = {r:.3f}, "
              f"p = {p_val:.4f} (n = {len(valid)})")
        fidelity_results.append({
            'Outcome': base,
            'r_fidelity': round(r, 4),
            'p_value': round(p_val, 6),
            'N': len(valid),
            'Significant_p05': p_val < 0.05,
        })
else:
    print("Warning: Implementation_Fidelity not "
          "available; skipping fidelity analysis.")

##### 5. Subgroup analyses by demographic variable
print("\n--- Subgroup Analyses (Treatment Group) ---")
subgroup_results = []
subgroup_vars = [
    v for v in [
        'Gender', 'Community', 'Education_Numeric']
    if v in treatment.columns]
for var in subgroup_vars:
    if treatment[var].nunique() > 6:
        continue
    for base in outcome_bases:
        change_col = f"{base}_Change"
        if change_col not in treatment.columns:
            continue
        sub = treatment[[var, change_col]].dropna()
        if sub[var].nunique() < 2:
            continue
        groups = [
            g[change_col].values
            for _, g in sub.groupby(var)]
        if len(groups) == 2:
            stat, p_val = stats.ttest_ind(*groups)
            test_type = 't_test'
        else:
            stat, p_val = stats.f_oneway(*groups)
            test_type = 'ANOVA'
        print(f"{base} by {var}: "
              f"{test_type} = {stat:.3f}, "
              f"p = {p_val:.4f}")
        subgroup_results.append({
            'Outcome': base,
            'Subgroup_Var': var,
            'Test_Type': test_type,
            'Statistic': round(stat, 4),
            'p_value': round(p_val, 6),
            'Significant_p05': p_val < 0.05,
        })

##### 6. Regression: predictors of outcome change
# (treatment group)
print("\n--- Regression: Predictors of Change "
      "(Treatment) ---")
regression_results = []
reg_predictors = [
    p for p in [
        'Age', 'Education_Numeric',
        'Implementation_Fidelity']
    if p in treatment.columns]
for base in outcome_bases:
    change_col = f"{base}_Change"
    if change_col not in treatment.columns:
        continue
    reg_data = treatment[
        reg_predictors + [change_col]].dropna()
    if len(reg_data) < len(reg_predictors) + 3:
        print(f"{base}: insufficient data for "
              f"regression.")
        continue
    formula = (
        f"`{change_col}` ~ "
        + " + ".join(reg_predictors))
    try:
        model = smf.ols(formula,
                        data=reg_data).fit()
        print(f"\n{base} ~ "
              f"{' + '.join(reg_predictors)}")
        print(f"  R2={model.rsquared:.4f}, "
              f"Adj R2 = {model.rsquared_adj:.4f}, "
              f"F({int(model.df_model)},"
              f"{int(model.df_resid)}) = "
              f"{model.fvalue:.3f}, "
              f"p={model.f_pvalue:.4f}")
        regression_results.append({
            'Outcome': base,
            'R2':        round(model.rsquared, 4),
            'Adj_R2':    round(
                model.rsquared_adj, 4),
            'F_stat':    round(model.fvalue, 4),
            'F_p_value': round(
                model.f_pvalue, 6),
            'N':         int(model.nobs),
        })
    except Exception as e:
        print(f"{base} regression failed: {e}")

##### 7. Export all results
pd.DataFrame(desc_records).to_csv(
    "out_F_evalua_langua_descriptives.csv",
    index=False)
all_inferential = (
    paired_results + between_results
    + fidelity_results + subgroup_results)
if all_inferential:
    pd.DataFrame(all_inferential).to_csv(
        "out_F_evalua_langua_inferential.csv",
        index=False)
if regression_results:
    pd.DataFrame(regression_results).to_csv(
        "out_F_evalua_langua_regression.csv",
        index=False)
print("\nEvaluating language interventions data analysis "
      "completed.")
print("Output files:")
print("  out_F_evalua_langua_descriptives.csv  "
      "-- descriptive statistics by group and time")
print("  out_F_evalua_langua_inferential.csv   "
      "-- t-tests, effect sizes, fidelity "
      "correlations, subgroup tests")
print("  out_F_evalua_langua_regression.csv    "
      "-- regression model fit statistics")
