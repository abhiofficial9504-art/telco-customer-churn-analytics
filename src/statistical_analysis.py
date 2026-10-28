"""
statistical_analysis.py
========================
Phase 9 — Bivariate and Multivariate Analysis with proper statistical tests.

- Numerical x Numerical: Pearson + Spearman correlation
- Categorical x Categorical: Chi-square + Cramer's V
- Categorical (binary target) x Numerical: point-biserial correlation
- Combined into one significance table with effect sizes and interpretation.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def cramers_v(confusion_matrix):
    """Bias-corrected Cramer's V for association between two categorical vars."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)


def effect_size_label_cramers_v(v):
    if pd.isna(v):
        return "N/A"
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Small"
    elif v < 0.5:
        return "Medium"
    else:
        return "Large"


def effect_size_label_corr(r):
    r = abs(r)
    if r < 0.1:
        return "Negligible"
    elif r < 0.3:
        return "Small"
    elif r < 0.5:
        return "Medium"
    else:
        return "Large"


def numeric_correlation_matrix(df):
    pearson = df[NUMERIC_COLS].corr(method="pearson")
    spearman = df[NUMERIC_COLS].corr(method="spearman")
    pearson.to_csv(os.path.join(TABLES_DIR, "correlation_pearson.csv"))
    spearman.to_csv(os.path.join(TABLES_DIR, "correlation_spearman.csv"))
    return pearson, spearman


def plot_correlation_heatmap(pearson, spearman):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pearson, annot=True, cmap="coolwarm", center=0, ax=axes[0], vmin=-1, vmax=1, fmt=".2f")
    axes[0].set_title("Pearson Correlation (Numeric Variables)")
    sns.heatmap(spearman, annot=True, cmap="coolwarm", center=0, ax=axes[1], vmin=-1, vmax=1, fmt=".2f")
    axes[1].set_title("Spearman Correlation (Numeric Variables)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "07_correlation_heatmaps.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def significance_table(df):
    """Build the Phase 9 statistical significance table:
    Feature | Test used | Test statistic | p-value | Effect size | Interpretation
    Target is always Churn (binary: Yes/No -> 1/0).
    """
    rows = []
    churn_binary = (df["Churn"] == "Yes").astype(int)

    # Categorical features -> Chi-square + Cramer's V against Churn
    for col in CATEGORICAL_COLS:
        ct = pd.crosstab(df[col], df["Churn"])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        v = cramers_v(ct)
        rows.append({
            "Feature": col,
            "Test Used": "Chi-square test of independence",
            "Test Statistic": round(chi2, 3),
            "Degrees of Freedom": dof,
            "p-value": p,
            "Effect Size (Cramer's V)": round(v, 3),
            "Effect Size Label": effect_size_label_cramers_v(v),
            "Significant (alpha=0.05)": "Yes" if p < 0.05 else "No",
            "Interpretation": (
                f"{'Statistically significant' if p < 0.05 else 'Not statistically significant'} "
                f"association with Churn; {effect_size_label_cramers_v(v).lower()} effect size."
            ),
        })

    # Numeric features -> point-biserial correlation with Churn (binary)
    for col in NUMERIC_COLS:
        r, p = stats.pointbiserialr(churn_binary, df[col])
        rows.append({
            "Feature": col,
            "Test Used": "Point-biserial correlation",
            "Test Statistic": round(r, 3),
            "Degrees of Freedom": len(df) - 2,
            "p-value": p,
            "Effect Size (Cramer's V)": np.nan,
            "Effect Size Label": effect_size_label_corr(r),
            "Significant (alpha=0.05)": "Yes" if p < 0.05 else "No",
            "Interpretation": (
                f"{'Statistically significant' if p < 0.05 else 'Not statistically significant'} "
                f"{'negative' if r < 0 else 'positive'} correlation with Churn; "
                f"{effect_size_label_corr(r).lower()} effect size."
            ),
        })

    table = pd.DataFrame(rows).sort_values("p-value")
    table["p-value"] = table["p-value"].apply(lambda x: f"{x:.2e}" if x < 0.0001 else round(x, 4))
    table.to_csv(os.path.join(TABLES_DIR, "statistical_significance_table.csv"), index=False)
    return table


def numeric_numeric_significance(df):
    """Pearson/Spearman p-values for the numeric-numeric pairs (supplementary)."""
    rows = []
    pairs = [("tenure", "MonthlyCharges"), ("tenure", "TotalCharges"), ("MonthlyCharges", "TotalCharges")]
    for a, b in pairs:
        r_p, p_p = stats.pearsonr(df[a], df[b])
        r_s, p_s = stats.spearmanr(df[a], df[b])
        rows.append({"Pair": f"{a} vs {b}", "Pearson r": round(r_p, 3), "Pearson p": p_p,
                      "Spearman rho": round(r_s, 3), "Spearman p": p_s})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES_DIR, "numeric_numeric_correlation_tests.csv"), index=False)
    return table


def crosstab_pivot_tables(df):
    """A few illustrative cross-tabs / pivot tables for the report."""
    ct1 = pd.crosstab(df["Contract"], df["InternetService"], values=df["Churn"].eq("Yes"), aggfunc="mean") * 100
    ct1 = ct1.round(2)
    ct1.to_csv(os.path.join(TABLES_DIR, "pivot_churnrate_contract_x_internet.csv"))

    ct2 = pd.crosstab(df["PaymentMethod"], df["Contract"], values=df["Churn"].eq("Yes"), aggfunc="mean") * 100
    ct2 = ct2.round(2)
    ct2.to_csv(os.path.join(TABLES_DIR, "pivot_churnrate_payment_x_contract.csv"))

    pivot_rev = df.pivot_table(index="Contract", columns="InternetService", values="MonthlyCharges",
                                aggfunc="mean").round(2)
    pivot_rev.to_csv(os.path.join(TABLES_DIR, "pivot_avgcharge_contract_x_internet.csv"))

    return ct1, ct2, pivot_rev


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "telco_features.csv"))
    pearson, spearman = numeric_correlation_matrix(df)
    print("Pearson correlation:\n", pearson.to_string())
    plot_correlation_heatmap(pearson, spearman)

    sig_table = significance_table(df)
    print("\nStatistical significance table (sorted by p-value):")
    print(sig_table.to_string())

    nn_table = numeric_numeric_significance(df)
    print("\nNumeric-numeric correlation significance:")
    print(nn_table.to_string())

    ct1, ct2, pivot_rev = crosstab_pivot_tables(df)
    print("\nChurn rate % pivot (Contract x InternetService):")
    print(ct1.to_string())
    print("\nChurn rate % pivot (PaymentMethod x Contract):")
    print(ct2.to_string())
