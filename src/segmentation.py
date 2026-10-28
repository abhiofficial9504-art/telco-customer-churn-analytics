"""
segmentation.py
================
Phase 10 — Customer Segmentation.

Builds business-defined (rule-based) segments — these are intentionally
overlapping targeting segments (as requested), not a mutually-exclusive
partition from clustering. Each segment is scored on size, churn rate,
tenure/value profile, and revenue contribution, then ranked by business
priority (defined as: MRR currently at risk = churned customers' MRR within
the segment, a direct, defensible dollar measure of "how much this segment
matters").
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")


def define_segments(df):
    median_mc = df["MonthlyCharges"].median()
    segments = {
        "New High-Risk Customers": (df["tenure"] <= 12) & (df["Contract"] == "Month-to-month"),
        "Loyal Long-Term Customers": (df["tenure"] >= 48) & (df["Contract"] != "Month-to-month"),
        "High-Value Loyal Customers": (df["HighValueCustomer"] == "Yes") & (df["tenure"] >= 24),
        "High-Value Churn Risk": (df["HighValueCustomer"] == "Yes") & (df["ContractRisk"] == "High"),
        "Low-Value Churn Risk": (df["HighValueCustomer"] == "No") & (df["ContractRisk"] == "High") &
                                 (df["MonthlyCharges"] < median_mc),
        "Fiber-Optic High-Risk Customers": (df["InternetService"] == "Fiber optic") &
                                            (df["ContractRisk"] == "High"),
        "Month-to-Month Risk Customers": (df["Contract"] == "Month-to-month"),
    }
    return segments


def segment_summary(df):
    segments = define_segments(df)
    total_customers = len(df)
    rows = []
    for name, mask in segments.items():
        sub = df[mask]
        n = len(sub)
        if n == 0:
            continue
        churn_rate = 100 * (sub["Churn"] == "Yes").mean()
        churned_sub = sub[sub["Churn"] == "Yes"]
        mrr_at_risk = churned_sub["MonthlyCharges"].sum()
        rows.append({
            "Segment": name,
            "Customer_Count": n,
            "Pct_of_Customer_Base_%": round(100 * n / total_customers, 2),
            "Churn_Rate_%": round(churn_rate, 2),
            "Avg_Tenure_Months": round(sub["tenure"].mean(), 1),
            "Avg_MonthlyCharges": round(sub["MonthlyCharges"].mean(), 2),
            "Avg_TotalCharges": round(sub["TotalCharges"].mean(), 2),
            "Segment_Total_MRR": round(sub["MonthlyCharges"].sum(), 2),
            "Pct_of_Total_MRR_%": round(100 * sub["MonthlyCharges"].sum() / df["MonthlyCharges"].sum(), 2),
            "MRR_At_Risk_from_Churned_in_Segment": round(mrr_at_risk, 2),
        })
    table = pd.DataFrame(rows)
    # Business priority: rank by $ MRR at risk (bigger dollar exposure = higher priority)
    table = table.sort_values("MRR_At_Risk_from_Churned_in_Segment", ascending=False).reset_index(drop=True)
    table.insert(0, "Priority_Rank", range(1, len(table) + 1))
    table.to_csv(os.path.join(TABLES_DIR, "customer_segment_summary.csv"), index=False)
    return table


def plot_segments(table):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = table.sort_values("MRR_At_Risk_from_Churned_in_Segment", ascending=True)
    axes[0].barh(order["Segment"], order["MRR_At_Risk_from_Churned_in_Segment"], color="#C0392B")
    axes[0].set_xlabel("Monthly Recurring Revenue At Risk ($)")
    axes[0].set_title("Top Churn-Risk Segments by Revenue Exposure")

    order2 = table.sort_values("Churn_Rate_%", ascending=True)
    colors = ["#C0392B" if c > 26.54 else "#2E86AB" for c in order2["Churn_Rate_%"]]
    axes[1].barh(order2["Segment"], order2["Churn_Rate_%"], color=colors)
    axes[1].axvline(26.54, color="black", linestyle="--", linewidth=1, label="Overall churn rate (26.5%)")
    axes[1].set_xlabel("Churn Rate (%)")
    axes[1].set_title("Segment Churn Rate vs. Company Average")
    axes[1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "10_top_risk_segments.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "telco_features.csv"))
    table = segment_summary(df)
    print("Customer segment summary (ranked by business priority / $ MRR at risk):")
    print(table.to_string())
    plot_segments(table)
