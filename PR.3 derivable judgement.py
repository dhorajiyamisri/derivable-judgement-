import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    ttest_ind,
    chi2_contingency,
    f_oneway,
    pearsonr
)

from statsmodels.stats.weightstats import ztest
from scipy.stats import t

# Genererate Dataset
np.random.seed(42)
n = 500

data = {
    
    "record_id": [f"R{i}" for i in range(1, n+1)],
    
    "age": np.random.randint(18, 70, n),
    
    "weight": np.random.randint(45, 100, n),
    
    "gender": np.random.choice(
        ["Male", "Female"],
        n
    ),
    
    "region": np.random.choice(
        ["North", "South", "East", "West"],
        n
    ),
    
    "smoking_status": np.random.choice(
        ["Smoker", "Non-Smoker", "Former Smoker"],
        n
    ),
    
    "exercise_frequency": np.random.choice(
        ["Daily", "Weekly", "Rarely", "Never"],
        n
    ),
    
    "bmi": np.random.normal(25, 4, n),
    
    "blood_pressure": np.random.normal(120, 15, n),
    
    "diabetes": np.random.choice(
        [True, False],
        n,
        p=[0.3, 0.7]
    ),
    
    "hypertension": np.random.choice(
        [True, False],
        n,
        p=[0.35, 0.65]
    ),
    
    "cholesterol_level": np.random.normal(200, 30, n),
    
    "glucose_level": np.random.normal(100, 20, n),
    
    "visit_date": pd.date_range(
        start="2025-01-01",
        periods=n,
        freq="D"
    )
}

df = pd.DataFrame(data)

# age group
df["age_group"] = pd.cut(
    df["age"],
    bins=[18,25,35,45,60,100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-60",
        "60+"
    ]
)

# Save Dataset
df.to_csv(
    "health_dataset.csv",
    index=False
)
print("Dataset Saved Successfully")

# Dataset Information
print(df.head())
print(df.info())
print(df.describe())

                  # PART A - THEORY NOTES
                  
# 1. Inferential Statistics:
# Used to make predictions about population using sample data.

# 2. Hypothesis Testing:
# Method to test assumptions using data.

# 3. Confidence Interval:
# Range in which true population value may exist.

# 4. P-value:
# Probability of getting results by chance.

# 5. Type I Error:
# Rejecting true null hypothesis.

# 6. Type II Error:
# Accepting false null hypothesis.

# 7. Z-test:
# Used when population SD is known.

# 8. T-test:
# Used to compare means.

# 9. Chi-square Test:
# Used for categorical data relationship.

# 10. ANOVA:
# Used to compare more than 2 groups.

# 11. Covariance:
# Measures direction of relationship.

# 12. Correlation:
# Measures strength of relationship.


                  # PART B - DATA ANALYSIS      
                  
# Hypothesis Examples
print("H0: Smoking has no effect on diabetes")
print("H1: Smoking affects diabetes prevalence")

# Confidence Interval
mean_bp = df["blood_pressure"].mean()
std_bp = df["blood_pressure"].std()
n = len(df)
confidence = 0.95
critical_value = t.ppf(
    (1 + confidence) / 2,
    df=n-1
)

margin_error = critical_value * (
    std_bp / np.sqrt(n)
)

lower = mean_bp - margin_error
upper = mean_bp + margin_error

print("Confidence Interval:")
print(lower, upper)

# Z-Test
z_stat, p_value = ztest(
    df["blood_pressure"],
    value=120
)

print("Z-Statistic =", z_stat)
print("P-Value =", p_value)   

# T-Test
male_bp = df[df["gender"]=="Male"]["blood_pressure"]
female_bp = df[df["gender"]=="Female"]["blood_pressure"]

t_stat, p_value = ttest_ind(
    male_bp,
    female_bp
)

print("T-Statistic =", t_stat)
print("P-Value =", p_value)

# Chi-Square Test
table = pd.crosstab(
    df["smoking_status"],
    df["diabetes"]
)

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-Square =", chi2)
print("P-Value =", p)

# ANOVA Test
g1 = df[df["age_group"]=="18-25"]["bmi"]
g2 = df[df["age_group"]=="26-35"]["bmi"]
g3 = df[df["age_group"]=="36-45"]["bmi"]

anova = f_oneway(g1, g2, g3)

print("ANOVA Result =", anova)

# Covariance
covariance = np.cov(
    df["age"],
    df["bmi"]
)

print(covariance)

# Correlation
corr, p = pearsonr(
    df["age"],
    df["bmi"]
)

print("Correlation =", corr)
print("P-value =", p)

# Histogram
sns.histplot(
    df["blood_pressure"],
    kde=True
)

plt.title("Blood Pressure Distribution")
plt.show()


# Boxplot
sns.boxplot(
    x="gender",
    y="blood_pressure",
    data=df
)

plt.title("Blood Pressure by Gender")
plt.show()

# Scatter Plot
sns.scatterplot(
    x="age",
    y="bmi",
    hue="gender",
    data=df
)

plt.title("Age vs BMI")
plt.show()                           