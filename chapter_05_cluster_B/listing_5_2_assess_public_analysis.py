import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# Load processed dataset (output of Listing 5.1)
df = pd.read_csv("out_B_assess_public_processed.csv")

##### Descriptive statistics
print("Descriptive Statistics for Attitude Score:")
print(df['Attitude_Score'].describe())

##### Histogram of attitude scores
plt.hist(df['Attitude_Score'], bins=10)
plt.title("Distribution of Public Attitude Scores")
plt.xlabel("Attitude Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

##### Group comparison example (Gender)
if 'Gender' in df.columns:
    group_means = (df.groupby('Gender')
                   ['Attitude_Score'].mean())
    print("\nMean Attitude Scores by Gender:")
    print(group_means)
    if df['Gender'].nunique() == 2:
        groups = [
            group["Attitude_Score"].values
            for name, group in df.groupby('Gender')]
        t_stat, p_val = stats.ttest_ind(*groups)
        print(f"\nT-test Results: t = {t_stat:.3f}, "
              f"p = {p_val:.3f}")
    else:
        print("Warning: Gender has more than 2 unique "
              "values. Skipping t-test.")
else:
    print("Warning: Gender column not found. "
          "Skipping group comparison.")

##### Regression model (example predictors)
predictors = []
if 'Age' in df.columns:
    predictors.append('Age')
if 'Education' in df.columns:
    predictors.append('Education')
if predictors:
    formula = ("Attitude_Score ~ "
                + " + ".join(predictors))
    model = smf.ols(formula, data=df).fit()
    print("\nRegression Results:")
    print(model.summary())
else:
    print("Warning: No valid predictor columns found. "
          "Skipping regression model.")

print("Public attitudes data analysis completed.")
