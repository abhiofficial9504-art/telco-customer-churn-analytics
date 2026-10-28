"""
features.py
============
Phase 11 — Advanced Feature Engineering.

Every derived feature is documented with its business interpretation.
This module is imported by every downstream analysis stage so that feature
definitions live in exactly one place (no duplicated/inconsistent logic).
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
# Columns that count as a "positive" add-on subscription when == 'Yes'
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]

FEATURE_DOCS = {}


def _doc(name, description):
    FEATURE_DOCS[name] = description


def engineer_features(df_in):
    """Return a new dataframe with all engineered features added. Does not
    mutate the input."""
    df = df_in.copy()

    # --- TenureGroup ---------------------------------------------------
    bins = [-0.1, 6, 12, 24, 36, 48, 60, np.inf]
    labels = ["0-6 months", "7-12 months", "13-24 months", "25-36 months",
              "37-48 months", "49-60 months", "61+ months"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    _doc("TenureGroup", "Customer lifecycle stage bucketed from tenure (months). Used to study "
                        "whether churn risk concentrates in early-life customers (onboarding risk) "
                        "vs. long-tenure customers (loyalty erosion).")

    # --- NumberOfServices (count of subscribed add-on/core services) ---
    def count_services(row):
        n = 0
        if row["PhoneService"] == "Yes":
            n += 1
        if row["MultipleLines"] == "Yes":
            n += 1
        if row["InternetService"] != "No":
            n += 1
        for c in ADDON_COLS[:-2] + ["StreamingTV", "StreamingMovies"] if False else ADDON_COLS:
            pass
        for c in ADDON_COLS:
            if row[c] == "Yes":
                n += 1
        return n

    df["NumberOfServices"] = df.apply(count_services, axis=1)
    _doc("NumberOfServices", "Count of active services/add-ons per customer (Phone, MultipleLines, "
                             "Internet [any type], OnlineSecurity, OnlineBackup, DeviceProtection, "
                             "TechSupport, StreamingTV, StreamingMovies). Range 0-9. Proxy for how "
                             "'embedded' / bundled a customer is with the provider.")

    # --- ServiceBundleSize (add-ons only, excludes phone/internet base) --
    df["ServiceBundleSize"] = df[ADDON_COLS].apply(lambda r: (r == "Yes").sum(), axis=1)
    _doc("ServiceBundleSize", "Count of the 6 optional add-on services subscribed (Online Security, "
                              "Online Backup, Device Protection, Tech Support, Streaming TV, Streaming "
                              "Movies). Range 0-6. Distinct from NumberOfServices, which also counts "
                              "the base Phone/Internet subscription.")

    # --- AverageChargePerMonth ------------------------------------------
    # For tenure=0 customers, avoid divide-by-zero; use MonthlyCharges as the estimate.
    df["AverageChargePerMonth"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    _doc("AverageChargePerMonth", "TotalCharges / tenure (fallback to current MonthlyCharges for "
                                  "brand-new tenure=0 customers). Smooths historical rate changes into "
                                  "a single lifetime-average monthly spend figure.")

    # --- HighMonthlyCharge (top quartile) --------------------------------
    q75 = df["MonthlyCharges"].quantile(0.75)
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] >= q75).map({True: "Yes", False: "No"})
    _doc("HighMonthlyCharge", f"Flag for customers in the top quartile of MonthlyCharges "
                              f"(>= {q75:.2f}). Used to isolate premium-paying customers for "
                              f"revenue-risk analysis.")

    # --- LongTermCustomer (tenure >= 24 months, i.e. 2 years) ------------
    df["LongTermCustomer"] = (df["tenure"] >= 24).map({True: "Yes", False: "No"})
    _doc("LongTermCustomer", "Flag for tenure >= 24 months (2 years). Splits the base into "
                             "established vs. newer relationships.")

    # --- HighValueCustomer (top quartile TotalCharges) -------------------
    q75_tc = df["TotalCharges"].quantile(0.75)
    df["HighValueCustomer"] = (df["TotalCharges"] >= q75_tc).map({True: "Yes", False: "No"})
    _doc("HighValueCustomer", f"Flag for customers in the top quartile of cumulative TotalCharges "
                              f"(>= {q75_tc:.2f}) — a simple lifetime-value proxy combining tenure "
                              f"and spend level.")

    # --- ContractRisk (categorical risk tier from contract type) ---------
    contract_risk_map = {"Month-to-month": "High", "One year": "Medium", "Two year": "Low"}
    df["ContractRisk"] = df["Contract"].map(contract_risk_map)
    _doc("ContractRisk", "Qualitative churn-risk tier derived directly from Contract type, based on "
                         "observed churn-rate ordering (Month-to-month=High, One year=Medium, "
                         "Two year=Low). Established empirically in the churn analysis, not assumed.")

    # --- PaymentRisk (categorical risk tier from payment method) ---------
    payment_risk_map = {
        "Electronic check": "High",
        "Mailed check": "Medium",
        "Bank transfer (automatic)": "Low",
        "Credit card (automatic)": "Low",
    }
    df["PaymentRisk"] = df["PaymentMethod"].map(payment_risk_map)
    _doc("PaymentRisk", "Qualitative churn-risk tier derived from PaymentMethod, based on observed "
                        "churn-rate ordering (Electronic check historically shows the highest churn; "
                        "automatic payments the lowest — automation likely reduces passive/frictional "
                        "churn from failed manual payments).")

    # --- CLV proxy (simple) ----------------------------------------------
    # Assumption stated explicitly: naive CLV = current MonthlyCharges x expected remaining lifetime.
    # We use two proxies: (a) realized CLV = TotalCharges (historical actual revenue), and
    # (b) a simple forward-looking proxy using contract-implied expected tenure.
    df["CLV_realized"] = df["TotalCharges"]
    _doc("CLV_realized", "Simple realized Customer Lifetime Value proxy = TotalCharges to date "
                         "(actual cumulative revenue collected). ASSUMPTION: does not discount cash "
                         "flows or net out service costs; it is a revenue-side proxy only.")

    # SeniorCitizen readable already handled upstream; ensure present
    if "SeniorCitizen_Flag" not in df.columns:
        df["SeniorCitizen_Flag"] = (df["SeniorCitizen"] == "Yes").astype(int)

    return df


def get_feature_documentation():
    return FEATURE_DOCS.copy()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "cleaned_telco.csv"))
    df_feat = engineer_features(df)
    df_feat.to_csv(os.path.join(TABLES_DIR, "telco_features.csv"), index=False)
    docs = get_feature_documentation()
    doc_df = pd.DataFrame([{"Feature": k, "Description": v} for k, v in docs.items()])
    doc_df.to_csv(os.path.join(TABLES_DIR, "feature_documentation.csv"), index=False)
    print(f"Engineered features. Shape: {df_feat.shape}")
    print(df_feat[["tenure", "TenureGroup", "NumberOfServices", "ServiceBundleSize",
                    "AverageChargePerMonth", "HighMonthlyCharge", "LongTermCustomer",
                    "HighValueCustomer", "ContractRisk", "PaymentRisk"]].head(10).to_string())
    print("\nFeature documentation:")
    for k, v in docs.items():
        print(f"- {k}: {v}")
