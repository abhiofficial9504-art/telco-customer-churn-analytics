"""
churn_analysis.py
==================
Phase 4  — Churn Analysis (overall + by every categorical/numeric driver)
Phase 5  — Tenure Analysis
Phase 6  — Revenue & Customer Value Analysis
Phase 7  — Service Adoption Analysis
Phase 8  — Contract & Payment Analysis (incl. risky combinations)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

CHURN_DRIVER_CATEGORICALS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

TENURE_BANDS = ["0-6 months", "7-12 months", "13-24 months", "25-36 months",
                "37-48 months", "49-60 months", "61+ months"]

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]


# ---------------------------------------------------------------------------
# PHASE 4 — CHURN ANALYSIS
# ---------------------------------------------------------------------------

def overall_churn_kpis(df):
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = total - churned
    kpis = {
        "Total Customers": total,
        "Churned Customers": int(churned),
        "Retained Customers": int(retained),
        "Churn Rate (%)": round(100 * churned / total, 2),
        "Retention Rate (%)": round(100 * retained / total, 2),
    }
    pd.DataFrame([kpis]).to_csv(os.path.join(TABLES_DIR, "churn_kpi_summary.csv"), index=False)
    return kpis


def churn_by_variable(df, col):
    """Per-category churn breakdown for one categorical variable."""
    total_churn = (df["Churn"] == "Yes").sum()
    grp = df.groupby(col, observed=True)
    out = grp.agg(
        Customer_Count=("Churn", "size"),
        Churn_Count=("Churn", lambda s: (s == "Yes").sum()),
    ).reset_index()
    out["Churn_Rate_%"] = round(100 * out["Churn_Count"] / out["Customer_Count"], 2)
    out["Retention_Rate_%"] = round(100 - out["Churn_Rate_%"], 2)
    out["Pct_of_Total_Churn_%"] = round(100 * out["Churn_Count"] / total_churn, 2)
    out["Variable"] = col
    out = out.rename(columns={col: "Category"})
    return out[["Variable", "Category", "Customer_Count", "Churn_Count", "Churn_Rate_%",
                "Retention_Rate_%", "Pct_of_Total_Churn_%"]]


def churn_by_all_categoricals(df):
    frames = [churn_by_variable(df, col) for col in CHURN_DRIVER_CATEGORICALS]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Churn_Rate_%", ascending=False).reset_index(drop=True)
    combined.to_csv(os.path.join(TABLES_DIR, "churn_by_category_all.csv"), index=False)
    return combined


def churn_by_numeric_bins(df):
    """Churn rate by quartile bins of MonthlyCharges and TotalCharges."""
    out_frames = []
    for col in ["MonthlyCharges", "TotalCharges"]:
        df_tmp = df.copy()
        df_tmp[f"{col}_Quartile"] = pd.qcut(df_tmp[col], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
        res = churn_by_variable(df_tmp.rename(columns={f"{col}_Quartile": "Category"}).assign(**{col: None}),
                                 "Category") if False else None
        grp = df_tmp.groupby(f"{col}_Quartile", observed=True)
        total_churn = (df["Churn"] == "Yes").sum()
        tab = grp.agg(Customer_Count=("Churn", "size"),
                       Churn_Count=("Churn", lambda s: (s == "Yes").sum())).reset_index()
        tab["Churn_Rate_%"] = round(100 * tab["Churn_Count"] / tab["Customer_Count"], 2)
        tab["Retention_Rate_%"] = round(100 - tab["Churn_Rate_%"], 2)
        tab["Pct_of_Total_Churn_%"] = round(100 * tab["Churn_Count"] / total_churn, 2)
        tab["Variable"] = col
        tab = tab.rename(columns={f"{col}_Quartile": "Category"})
        out_frames.append(tab[["Variable", "Category", "Customer_Count", "Churn_Count", "Churn_Rate_%",
                                "Retention_Rate_%", "Pct_of_Total_Churn_%"]])
    combined = pd.concat(out_frames, ignore_index=True)
    combined.to_csv(os.path.join(TABLES_DIR, "churn_by_numeric_quartile.csv"), index=False)
    return combined


def identify_strongest_risk_groups(combined_categorical, min_n=100, top_n=10):
    """Highest-risk groups (excluding tiny/near-100% artifact groups like
    'No internet service' which are structurally 0% churn by definition,
    and requiring a minimum sample size for reliability)."""
    filtered = combined_categorical[combined_categorical["Customer_Count"] >= min_n]
    top_risk = filtered.sort_values("Churn_Rate_%", ascending=False).head(top_n)
    top_risk.to_csv(os.path.join(TABLES_DIR, "top_churn_risk_groups.csv"), index=False)
    return top_risk


# ---------------------------------------------------------------------------
# PHASE 5 — TENURE ANALYSIS
# ---------------------------------------------------------------------------

def tenure_band_analysis(df):
    total_churn = (df["Churn"] == "Yes").sum()
    grp = df.groupby("TenureGroup", observed=True)
    tab = grp.agg(
        Customer_Count=("Churn", "size"),
        Churn_Count=("Churn", lambda s: (s == "Yes").sum()),
        Avg_MonthlyCharges=("MonthlyCharges", "mean"),
        Avg_TotalCharges=("TotalCharges", "mean"),
        Pct_MonthToMonth=("Contract", lambda s: 100 * (s == "Month-to-month").mean()),
        Avg_NumberOfServices=("NumberOfServices", "mean"),
    ).reset_index()
    tab["Churn_Rate_%"] = round(100 * tab["Churn_Count"] / tab["Customer_Count"], 2)
    tab["Pct_of_Customer_Base_%"] = round(100 * tab["Customer_Count"] / len(df), 2)
    tab["Pct_of_Total_Churn_%"] = round(100 * tab["Churn_Count"] / total_churn, 2)
    tab["Avg_MonthlyCharges"] = tab["Avg_MonthlyCharges"].round(2)
    tab["Avg_TotalCharges"] = tab["Avg_TotalCharges"].round(2)
    tab["Pct_MonthToMonth"] = tab["Pct_MonthToMonth"].round(1)
    tab["Avg_NumberOfServices"] = tab["Avg_NumberOfServices"].round(2)
    tab["TenureGroup"] = pd.Categorical(tab["TenureGroup"], categories=TENURE_BANDS, ordered=True)
    tab = tab.sort_values("TenureGroup").reset_index(drop=True)
    tab.to_csv(os.path.join(TABLES_DIR, "tenure_band_analysis.csv"), index=False)
    return tab


def plot_tenure_churn(tenure_tab):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tenure_tab))
    bars = ax1.bar(x, tenure_tab["Customer_Count"], color="#8FBFE0", label="Customer Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tenure_tab["TenureGroup"], rotation=30, ha="right")
    ax1.set_ylabel("Customer Count")
    ax1.set_title("Churn Rate by Tenure Band (bars = customer count, line = churn rate)")
    ax2 = ax1.twinx()
    ax2.plot(x, tenure_tab["Churn_Rate_%"], color="#C0392B", marker="o", linewidth=2.5, label="Churn Rate (%)")
    ax2.set_ylabel("Churn Rate (%)")
    for i, v in enumerate(tenure_tab["Churn_Rate_%"]):
        ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 8), ha="center",
                      color="#C0392B", fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_churn_by_tenure_band.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# PHASE 6 — REVENUE & CUSTOMER VALUE ANALYSIS
# ---------------------------------------------------------------------------

def revenue_analysis(df):
    total_mrr = df["MonthlyCharges"].sum()
    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]

    revenue_at_risk_monthly = churned["MonthlyCharges"].sum()
    revenue_at_risk_annualized = revenue_at_risk_monthly * 12

    summary = {
        "Total Monthly Recurring Revenue (MRR)": round(total_mrr, 2),
        "Average Monthly Revenue per Customer": round(df["MonthlyCharges"].mean(), 2),
        "MRR from Churned Customers": round(revenue_at_risk_monthly, 2),
        "MRR from Retained Customers": round(retained["MonthlyCharges"].sum(), 2),
        "Estimated Annualized Revenue at Risk (churned MRR x 12)": round(revenue_at_risk_annualized, 2),
        "% of Total MRR at Risk": round(100 * revenue_at_risk_monthly / total_mrr, 2),
        "Avg TotalCharges (all)": round(df["TotalCharges"].mean(), 2),
        "Avg TotalCharges (churned)": round(churned["TotalCharges"].mean(), 2),
        "Avg TotalCharges (retained)": round(retained["TotalCharges"].mean(), 2),
        "Avg MonthlyCharges (churned)": round(churned["MonthlyCharges"].mean(), 2),
        "Avg MonthlyCharges (retained)": round(retained["MonthlyCharges"].mean(), 2),
        "Avg tenure (churned)": round(churned["tenure"].mean(), 2),
        "Avg tenure (retained)": round(retained["tenure"].mean(), 2),
        "Avg CLV_realized (churned)": round(churned["CLV_realized"].mean(), 2),
        "Avg CLV_realized (retained)": round(retained["CLV_realized"].mean(), 2),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(TABLES_DIR, "revenue_summary.csv"), index=False)

    # High-value churners: top quartile TotalCharges AND churned
    high_value_churners = churned[churned["HighValueCustomer"] == "Yes"]
    long_tenure_churners = churned[churned["tenure"] >= 24]
    high_mc_at_risk = df[(df["HighMonthlyCharge"] == "Yes")]
    high_mc_churn_rate = round(100 * (high_mc_at_risk["Churn"] == "Yes").mean(), 2)

    deep_dive = {
        "High-value churners (top-quartile TotalCharges & churned)": len(high_value_churners),
        "  -> their combined lost MRR": round(high_value_churners["MonthlyCharges"].sum(), 2),
        "Long-tenure churners (tenure>=24mo & churned)": len(long_tenure_churners),
        "  -> their combined lost MRR": round(long_tenure_churners["MonthlyCharges"].sum(), 2),
        "Churn rate among HighMonthlyCharge customers (top quartile)": high_mc_churn_rate,
    }

    # Revenue concentration: what % of MRR comes from top 25% of spenders?
    df_sorted = df.sort_values("MonthlyCharges", ascending=False)
    top25_n = int(len(df_sorted) * 0.25)
    top25_mrr_share = round(100 * df_sorted.head(top25_n)["MonthlyCharges"].sum() / total_mrr, 2)
    deep_dive["Revenue concentration: % of MRR from top 25% of customers by MonthlyCharges"] = top25_mrr_share

    with open(os.path.join(REPORTS_DIR, "phase6_revenue_deepdive.txt"), "w") as f:
        for k, v in deep_dive.items():
            f.write(f"{k}: {v}\n")

    return summary, deep_dive


def plot_revenue_at_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    churned = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()
    retained = df[df["Churn"] == "No"]["MonthlyCharges"].sum()
    axes[0].pie([churned, retained], labels=[f"Churned MRR\n${churned:,.0f}", f"Retained MRR\n${retained:,.0f}"],
                autopct="%1.1f%%", colors=["#C0392B", "#2E86AB"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Monthly Recurring Revenue: At Risk vs. Retained")

    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1], palette={"Yes": "#C0392B", "No": "#2E86AB"})
    axes[1].set_title("MonthlyCharges Distribution by Churn Status")
    axes[1].set_xlabel("Churn")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "09_revenue_at_risk.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# PHASE 7 — SERVICE ADOPTION ANALYSIS
# ---------------------------------------------------------------------------

def service_adoption_analysis(df):
    rows = []
    service_cols = ["PhoneService", "MultipleLines"] + ADDON_COLS
    for col in service_cols:
        users = df[df[col] == "Yes"]
        non_users = df[df[col] == "No"]
        penetration = round(100 * len(users) / len(df), 2)
        churn_users = round(100 * (users["Churn"] == "Yes").mean(), 2) if len(users) else np.nan
        churn_non_users = round(100 * (non_users["Churn"] == "Yes").mean(), 2) if len(non_users) else np.nan
        rows.append({
            "Service": col,
            "Penetration_Rate_%": penetration,
            "Users_N": len(users),
            "Churn_Rate_Among_Users_%": churn_users,
            "Non_Users_N": len(non_users),
            "Churn_Rate_Among_NonUsers_%": churn_non_users,
            "Churn_Rate_Difference_pp": round(churn_non_users - churn_users, 2) if pd.notna(churn_users) else np.nan,
        })
    table = pd.DataFrame(rows).sort_values("Churn_Rate_Difference_pp", ascending=False)
    table.to_csv(os.path.join(TABLES_DIR, "service_adoption_analysis.csv"), index=False)

    # Churn by NumberOfServices
    nos_grp = df.groupby("NumberOfServices").agg(
        Customer_Count=("Churn", "size"),
        Churn_Rate_pct=("Churn", lambda s: round(100 * (s == "Yes").mean(), 2)),
    ).reset_index()
    nos_grp.to_csv(os.path.join(TABLES_DIR, "churn_by_number_of_services.csv"), index=False)

    return table, nos_grp


def plot_service_adoption(service_tab, nos_grp):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = service_tab.sort_values("Churn_Rate_Difference_pp", ascending=True)
    y = np.arange(len(order))
    axes[0].barh(y, order["Churn_Rate_Among_Users_%"], color="#C0392B", label="Users churn %", alpha=0.85)
    axes[0].barh(y, order["Churn_Rate_Among_NonUsers_%"], color="#2E86AB", label="Non-users churn %", alpha=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order["Service"])
    axes[0].set_xlabel("Churn Rate (%)")
    axes[0].set_title("Churn Rate: Service Users vs. Non-Users")
    axes[0].legend()

    axes[1].bar(nos_grp["NumberOfServices"].astype(str), nos_grp["Churn_Rate_pct"], color="#55A868")
    axes[1].set_xlabel("Number of Services Subscribed")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate vs. Number of Services (Bundle Depth)")
    for i, v in enumerate(nos_grp["Churn_Rate_pct"]):
        axes[1].annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "11_service_adoption.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# PHASE 8 — CONTRACT & PAYMENT ANALYSIS
# ---------------------------------------------------------------------------

def contract_payment_analysis(df):
    contract_tab = df.groupby("Contract", observed=True).agg(
        Customer_Count=("Churn", "size"),
        Churn_Rate_pct=("Churn", lambda s: round(100 * (s == "Yes").mean(), 2)),
        Avg_MonthlyCharges=("MonthlyCharges", "mean"),
        Avg_tenure=("tenure", "mean"),
    ).round(2).reset_index()
    contract_tab.to_csv(os.path.join(TABLES_DIR, "churn_by_contract.csv"), index=False)

    payment_tab = df.groupby("PaymentMethod", observed=True).agg(
        Customer_Count=("Churn", "size"),
        Churn_Rate_pct=("Churn", lambda s: round(100 * (s == "Yes").mean(), 2)),
        Avg_MonthlyCharges=("MonthlyCharges", "mean"),
    ).round(2).reset_index().sort_values("Churn_Rate_pct", ascending=False)
    payment_tab.to_csv(os.path.join(TABLES_DIR, "churn_by_payment_method.csv"), index=False)

    paperless_tab = df.groupby("PaperlessBilling", observed=True).agg(
        Customer_Count=("Churn", "size"),
        Churn_Rate_pct=("Churn", lambda s: round(100 * (s == "Yes").mean(), 2)),
    ).round(2).reset_index()
    paperless_tab.to_csv(os.path.join(TABLES_DIR, "churn_by_paperless_billing.csv"), index=False)

    # Risky combinations
    combos = []

    def combo_stat(name, mask):
        sub = df[mask]
        if len(sub) == 0:
            return
        combos.append({
            "Combination": name,
            "Customer_Count": len(sub),
            "Pct_of_Base_%": round(100 * len(sub) / len(df), 2),
            "Churn_Rate_%": round(100 * (sub["Churn"] == "Yes").mean(), 2),
            "Avg_MonthlyCharges": round(sub["MonthlyCharges"].mean(), 2),
        })

    combo_stat("Month-to-month + Electronic check",
               (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check"))
    combo_stat("Month-to-month + Fiber optic",
               (df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic"))
    combo_stat("Month-to-month + Paperless billing",
               (df["Contract"] == "Month-to-month") & (df["PaperlessBilling"] == "Yes"))
    combo_stat("High MonthlyCharge (top quartile) + Month-to-month",
               (df["HighMonthlyCharge"] == "Yes") & (df["Contract"] == "Month-to-month"))
    combo_stat("Month-to-month + Fiber optic + Electronic check",
               (df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic") &
               (df["PaymentMethod"] == "Electronic check"))
    combo_stat("Two year + Automatic payment (bank/credit)",
               (df["Contract"] == "Two year") & (df["PaymentMethod"].isin(
                   ["Bank transfer (automatic)", "Credit card (automatic)"])))
    combo_stat("Month-to-month + No online security + Fiber optic",
               (df["Contract"] == "Month-to-month") & (df["OnlineSecurity"] == "No") &
               (df["InternetService"] == "Fiber optic"))
    combo_stat("New customer (tenure<=6mo) + Month-to-month",
               (df["tenure"] <= 6) & (df["Contract"] == "Month-to-month"))

    combo_tab = pd.DataFrame(combos).sort_values("Churn_Rate_%", ascending=False)
    combo_tab.to_csv(os.path.join(TABLES_DIR, "risky_combinations.csv"), index=False)

    return contract_tab, payment_tab, paperless_tab, combo_tab


def plot_contract_payment(contract_tab, payment_tab, combo_tab):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    axes[0].bar(contract_tab["Contract"], contract_tab["Churn_Rate_pct"], color="#DD8452", width=0.6)
    axes[0].set_title("Churn Rate by Contract Type")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_ylim(0, contract_tab["Churn_Rate_pct"].max() * 1.2)
    axes[0].tick_params(axis="x", labelsize=10)
    for i, v in enumerate(contract_tab["Churn_Rate_pct"]):
        axes[0].annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom")

    axes[1].barh(payment_tab["PaymentMethod"], payment_tab["Churn_Rate_pct"], color="#4C72B0")
    axes[1].set_title("Churn Rate by Payment Method")
    axes[1].set_xlabel("Churn Rate (%)")
    for i, v in enumerate(payment_tab["Churn_Rate_pct"]):
        axes[1].annotate(f"{v:.1f}%", (v, i), va="center", fontsize=9)

    top_combos = combo_tab.sort_values("Churn_Rate_%", ascending=True).tail(8)
    axes[2].barh(top_combos["Combination"], top_combos["Churn_Rate_%"], color="#C0392B")
    axes[2].set_title("Highest-Risk Contract x Payment x Service\nCombinations")
    axes[2].set_xlabel("Churn Rate (%)")
    for i, v in enumerate(top_combos["Churn_Rate_%"]):
        axes[2].annotate(f"{v:.1f}%", (v, i), va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_churn_by_contract_payment.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_churn_kpi_and_distribution(df, kpis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    counts = [kpis["Retained Customers"], kpis["Churned Customers"]]
    axes[0].pie(counts, labels=[f"Retained\n{counts[0]:,} ({kpis['Retention Rate (%)']}%)",
                                 f"Churned\n{counts[1]:,} ({kpis['Churn Rate (%)']}%)"],
                colors=["#2E86AB", "#C0392B"], autopct=None, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Overall Customer Churn Distribution")

    internet_tab = df.groupby("InternetService", observed=True)["Churn"].apply(
        lambda s: 100 * (s == "Yes").mean()).sort_values(ascending=False)
    axes[1].bar(internet_tab.index, internet_tab.values, color="#DD8452")
    axes[1].set_title("Churn Rate by Internet Service Type")
    axes[1].set_ylabel("Churn Rate (%)")
    for i, v in enumerate(internet_tab.values):
        axes[1].annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "06_churn_kpi_and_internet.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "telco_features.csv"))

    kpis = overall_churn_kpis(df)
    print("Churn KPIs:", kpis)

    combined_cat = churn_by_all_categoricals(df)
    print("\nTop 10 highest-churn categories (all, incl. small groups):")
    print(combined_cat.head(10).to_string())

    top_risk = identify_strongest_risk_groups(combined_cat)
    print("\nStrongest risk groups (n>=100):")
    print(top_risk.to_string())

    num_bins = churn_by_numeric_bins(df)
    print("\nChurn by charge quartile:")
    print(num_bins.to_string())

    tenure_tab = tenure_band_analysis(df)
    print("\nTenure band analysis:")
    print(tenure_tab.to_string())
    plot_tenure_churn(tenure_tab)

    rev_summary, rev_deepdive = revenue_analysis(df)
    print("\nRevenue summary:", rev_summary)
    print("\nRevenue deep dive:", rev_deepdive)
    plot_revenue_at_risk(df)

    service_tab, nos_grp = service_adoption_analysis(df)
    print("\nService adoption:")
    print(service_tab.to_string())
    print("\nChurn by NumberOfServices:")
    print(nos_grp.to_string())
    plot_service_adoption(service_tab, nos_grp)

    contract_tab, payment_tab, paperless_tab, combo_tab = contract_payment_analysis(df)
    print("\nContract:", contract_tab.to_string())
    print("\nPayment:", payment_tab.to_string())
    print("\nPaperless:", paperless_tab.to_string())
    print("\nRisky combos:")
    print(combo_tab.to_string())
    plot_contract_payment(contract_tab, payment_tab, combo_tab)
    plot_churn_kpi_and_distribution(df, kpis)

    print("\nAll Phase 4-8 outputs written to outputs/tables and outputs/figures.")
