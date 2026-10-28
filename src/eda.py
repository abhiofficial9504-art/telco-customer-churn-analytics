"""
eda.py
======
Phase 3 — Univariate Analysis (numerical + categorical), with charts and
short analytical interpretations for each.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
os.makedirs(FIG_DIR, exist_ok=True)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]

INSIGHTS = []  # collects (chart_file, interpretation) pairs


def numeric_summary_table(df):
    rows = []
    for col in NUMERIC_COLS:
        s = df[col].dropna()
        desc = s.describe()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        rows.append({
            "Variable": col,
            "Count": int(desc["count"]),
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "Mode": round(s.mode().iloc[0], 2) if not s.mode().empty else np.nan,
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "Range": round(s.max() - s.min(), 2),
            "Std Dev": round(s.std(), 2),
            "Variance": round(s.var(), 2),
            "Q1 (25%)": round(q1, 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Skewness": round(stats.skew(s), 3),
            "Kurtosis": round(stats.kurtosis(s), 3),
            "P5": round(s.quantile(0.05), 2),
            "P95": round(s.quantile(0.95), 2),
            "Coeff of Variation (%)": round(100 * s.std() / s.mean(), 2),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES_DIR, "numeric_univariate_summary.csv"), index=False)
    return table


def categorical_summary_table(df):
    rows = []
    for col in CATEGORICAL_COLS:
        vc = df[col].value_counts()
        pct = df[col].value_counts(normalize=True) * 100
        top_cat = vc.index[0]
        rare_cats = [c for c in vc.index if pct[c] < 5]
        rows.append({
            "Variable": col,
            "Unique Categories": df[col].nunique(),
            "Top Category": top_cat,
            "Top Category %": round(pct.iloc[0], 2),
            "Rare Categories (<5%)": ", ".join(rare_cats) if rare_cats else "None",
            "Full Distribution": "; ".join([f"{k}={v} ({pct[k]:.1f}%)" for k, v in vc.items()]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES_DIR, "categorical_univariate_summary.csv"), index=False)
    return table


def plot_numeric_distributions(df):
    """Histogram + KDE + boxplot grid for the three numeric variables."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    for i, col in enumerate(NUMERIC_COLS):
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color="#4C72B0", bins=30)
        axes[i, 0].set_title(f"{col} — Distribution (Histogram + KDE)")
        axes[i, 0].set_xlabel(col)
        sns.boxplot(x=df[col], ax=axes[i, 1], color="#DD8452")
        axes[i, 1].set_title(f"{col} — Boxplot")
        axes[i, 1].set_xlabel(col)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_numeric_distributions.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_numeric_violin(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(NUMERIC_COLS):
        sns.violinplot(y=df[col], ax=axes[i], color="#55A868", inner="quartile")
        axes[i].set_title(f"{col} — Violin Plot")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_numeric_violin.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_categorical_bars(df):
    """Bar charts for all categorical variables (grid), with % labels."""
    cols = [c for c in CATEGORICAL_COLS if c != "Churn"]  # Churn covered in Phase 4
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.6))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        vc = df[col].value_counts()
        pct = df[col].value_counts(normalize=True) * 100
        bars = axes[i].bar(vc.index.astype(str), vc.values, color=sns.color_palette("Set2")[i % 8])
        axes[i].set_title(col, fontsize=11)
        axes[i].tick_params(axis="x", rotation=30, labelsize=8)
        for bar, p in zip(bars, pct.values):
            axes[i].annotate(f"{p:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              ha="center", va="bottom", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Categorical Variable Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_categorical_distributions.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_interpretations(df, num_table, cat_table):
    """Business-language interpretation of each univariate finding."""
    tenure_skew = num_table.loc[num_table.Variable == "tenure", "Skewness"].iloc[0]
    mc_skew = num_table.loc[num_table.Variable == "MonthlyCharges", "Skewness"].iloc[0]
    tc_skew = num_table.loc[num_table.Variable == "TotalCharges", "Skewness"].iloc[0]

    notes = []
    notes.append(
        f"Tenure distribution (skewness={tenure_skew:.2f}) is close to a bimodal/uniform-ish shape spanning "
        f"0-72 months with a spike at very low tenure — a large group of recently-acquired customers "
        f"coexists with a substantial base of long-tenured (5-6 year) customers. This bimodality is a "
        f"strong early signal that acquisition and retention dynamics may differ sharply by lifecycle stage."
    )
    notes.append(
        f"MonthlyCharges (skewness={mc_skew:.2f}) is roughly symmetric with a slight left lean, driven by "
        f"a visible cluster near the low end (~$20, single-service/no-internet customers) and a broad hump "
        f"in the $70-$110 range (fiber + add-on bundles) — the market is effectively two customer "
        f"populations at different price points, not one continuum."
    )
    notes.append(
        f"TotalCharges (skewness={tc_skew:.2f}) is strongly right-skewed, as expected for a cumulative "
        f"billing variable — most customers have low cumulative spend (short tenure), while a smaller "
        f"group of long-tenured customers has accumulated very high lifetime billing."
    )
    top_internet = df["InternetService"].value_counts(normalize=True) * 100
    notes.append(
        f"InternetService: Fiber optic is the single largest segment "
        f"({top_internet.get('Fiber optic', 0):.1f}% of customers), ahead of DSL "
        f"({top_internet.get('DSL', 0):.1f}%) — fiber pricing/quality will disproportionately shape "
        f"overall churn and revenue dynamics."
    )
    top_contract = df["Contract"].value_counts(normalize=True) * 100
    notes.append(
        f"Contract: Month-to-month is the dominant contract type ({top_contract.get('Month-to-month', 0):.1f}% "
        f"of the base) versus only {top_contract.get('Two year', 0):.1f}% on two-year contracts — the customer "
        f"base skews heavily toward the most flexible (and, as later analysis shows, the highest-churn-risk) "
        f"commitment tier."
    )
    top_payment = df["PaymentMethod"].value_counts(normalize=True) * 100
    notes.append(
        f"PaymentMethod: Electronic check is the most common method ({top_payment.get('Electronic check', 0):.1f}%), "
        f"notably more common than either automatic payment method — a meaningful share of the base is on a "
        f"manual, non-automated billing method, which (per churn analysis) correlates with higher churn."
    )
    senior_pct = (df["SeniorCitizen"] == "Yes").mean() * 100
    notes.append(
        f"SeniorCitizen: Only {senior_pct:.1f}% of customers are senior citizens — a small but (per churn "
        f"analysis) disproportionately high-risk subgroup worth targeted attention despite its small size."
    )
    return notes


def run_eda(df):
    num_table = numeric_summary_table(df)
    cat_table = categorical_summary_table(df)
    fig1 = plot_numeric_distributions(df)
    fig2 = plot_numeric_violin(df)
    fig3 = plot_categorical_bars(df)
    interpretations = generate_interpretations(df, num_table, cat_table)
    with open(os.path.join(BASE_DIR, "outputs", "reports", "phase3_eda_interpretations.txt"), "w") as f:
        f.write("\n\n".join(interpretations))
    return {
        "numeric_table": num_table,
        "categorical_table": cat_table,
        "figures": [fig1, fig2, fig3],
        "interpretations": interpretations,
    }


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "telco_features.csv"))
    results = run_eda(df)
    print("Numeric summary:")
    print(results["numeric_table"].to_string())
    print("\nCategorical summary (first 5):")
    print(results["categorical_table"].drop(columns=["Full Distribution"]).head().to_string())
    print("\nFigures generated:", results["figures"])
    print("\nInterpretations:")
    for n in results["interpretations"]:
        print("-", n)
