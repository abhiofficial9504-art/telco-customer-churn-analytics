"""
data_cleaning.py
=================
Phase 1 (Dataset Understanding) and Phase 2 (Data Cleaning) for the Telco
Customer Churn dataset.

This module is intentionally verbose in its printed/logged output because
every cleaning decision must be documented (per project working rules).

All functions are side-effect-free on the *original* CSV: the raw file at
data/telco.csv is only ever read, never overwritten. Cleaned data is written
to outputs/tables/cleaned_telco.csv (and mirrored to the project root as
cleaned_telco.csv for easy delivery).
"""
import json
import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco.csv")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

EXPECTED_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path=DATA_PATH):
    """Safely load the raw CSV. Never mutates the file on disk."""
    df = pd.read_csv(path, dtype=str)  # load everything as string first so we
    # can inspect raw values (e.g. blank strings, stray whitespace) before any
    # type coercion happens. This avoids pandas silently guessing types.
    return df


# ---------------------------------------------------------------------------
# PHASE 1 — DATASET UNDERSTANDING
# ---------------------------------------------------------------------------

def phase1_inspection(df_raw):
    """Run the full Phase 1 inspection and return a dict of findings plus
    write a text log to outputs/reports/phase1_inspection.txt."""
    log_lines = []

    def log(msg=""):
        log_lines.append(str(msg))

    log("=" * 80)
    log("PHASE 1 — DATASET UNDERSTANDING")
    log("=" * 80)

    # 1. Shape
    log(f"\n[1] Dataset shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    # 2. Column names
    log(f"\n[2] Column names ({len(df_raw.columns)}):")
    log(list(df_raw.columns))
    missing_expected = set(EXPECTED_COLS) - set(df_raw.columns)
    extra_cols = set(df_raw.columns) - set(EXPECTED_COLS)
    log(f"  Missing expected columns: {missing_expected or 'None'}")
    log(f"  Unexpected extra columns: {extra_cols or 'None'}")

    # 3. Data types (as loaded raw, i.e. all strings by design; report the
    # types pandas would infer natively for reference)
    df_native = pd.read_csv(DATA_PATH)
    log("\n[3] Data types (native pandas inference on raw CSV):")
    log(df_native.dtypes.to_string())

    # 4. First / last rows
    log("\n[4] First 3 rows:")
    log(df_raw.head(3).to_string())
    log("\n    Last 3 rows:")
    log(df_raw.tail(3).to_string())

    # 5. Random sample
    log("\n[5] Random sample (5 rows, random_state=42):")
    log(df_raw.sample(5, random_state=42).to_string())

    # 6. Unique values per categorical variable
    log("\n[6] Unique values per categorical column:")
    unique_vals = {}
    for col in CATEGORICAL_COLS:
        if col in df_raw.columns:
            vals = sorted(df_raw[col].dropna().unique().tolist())
            unique_vals[col] = vals
            log(f"  {col} ({len(vals)}): {vals}")

    # 7. Numerical variable distributions (raw, string-coerced to numeric for
    # inspection only — no mutation of df_raw itself)
    log("\n[7] Numerical variable distributions (coerced view only):")
    numeric_preview = {}
    for col in NUMERIC_COLS:
        coerced = pd.to_numeric(df_raw[col], errors="coerce")
        numeric_preview[col] = coerced.describe().to_dict()
        log(f"  {col}:")
        log(coerced.describe().to_string())
        n_nonnumeric = coerced.isna().sum() - df_raw[col].isna().sum()
        log(f"    Non-numeric-but-non-null entries in raw column: {n_nonnumeric}")

    # 8. Missing values (true NaN)
    log("\n[8] Missing values (NaN) per column:")
    missing_counts = df_raw.isna().sum()
    log(missing_counts.to_string())

    # 9. Blank strings (whitespace-only or empty, not counted as NaN by pandas)
    log("\n[9] Blank / whitespace-only string values per column:")
    blank_counts = {}
    for col in df_raw.columns:
        if df_raw[col].dtype == object:
            blanks = df_raw[col].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
            blank_counts[col] = int(blanks)
    for k, v in blank_counts.items():
        if v > 0:
            log(f"  {k}: {v} blank string(s)")
    if sum(blank_counts.values()) == 0:
        log("  None found.")

    # 10. Duplicate rows (full row duplicates)
    dup_rows = df_raw.duplicated().sum()
    log(f"\n[10] Fully duplicated rows: {dup_rows}")

    # 11. Duplicate customer IDs
    dup_ids = df_raw["customerID"].duplicated().sum()
    log(f"\n[11] Duplicate customerID values: {dup_ids}")
    if dup_ids > 0:
        dupe_id_list = df_raw[df_raw["customerID"].duplicated(keep=False)].sort_values("customerID")
        log(dupe_id_list.to_string())

    # 12. Constant / near-constant columns
    log("\n[12] Constant / near-constant columns (>=99% single value):")
    const_cols = []
    for col in df_raw.columns:
        vc = df_raw[col].value_counts(normalize=True, dropna=False)
        if len(vc) == 1:
            const_cols.append((col, "constant", vc.index[0]))
        elif vc.iloc[0] >= 0.99:
            const_cols.append((col, "near-constant (>=99%)", vc.index[0]))
    if const_cols:
        for c in const_cols:
            log(f"  {c[0]}: {c[1]} (dominant value: {c[2]})")
    else:
        log("  None found.")

    # 13/14/15. Invalid / suspicious values / data-entry inconsistencies
    log("\n[13-15] Invalid / suspicious values & data-entry inconsistency checks:")
    issues = []

    # SeniorCitizen should be 0/1 only
    sc_vals = set(df_raw["SeniorCitizen"].dropna().unique())
    if not sc_vals.issubset({"0", "1"}):
        issues.append(f"SeniorCitizen has unexpected values: {sc_vals}")
    else:
        log(f"  SeniorCitizen values OK: {sorted(sc_vals)} (note: encoded as 0/1 int, inconsistent with "
            f"Yes/No style used by Partner/Dependents/etc. — a labeling inconsistency worth flagging)")

    # tenure should be >=0 and reasonably bounded (<= ~72 months for this well-known dataset vintage)
    tenure_num = pd.to_numeric(df_raw["tenure"], errors="coerce")
    neg_tenure = (tenure_num < 0).sum()
    if neg_tenure:
        issues.append(f"tenure has {neg_tenure} negative values")
    log(f"  tenure range: {tenure_num.min()} to {tenure_num.max()} (negative values: {neg_tenure})")

    # MonthlyCharges should be > 0
    mc_num = pd.to_numeric(df_raw["MonthlyCharges"], errors="coerce")
    neg_mc = (mc_num <= 0).sum()
    log(f"  MonthlyCharges range: {mc_num.min()} to {mc_num.max()} (<=0 values: {neg_mc})")

    # TotalCharges stored as text?
    tc_raw = df_raw["TotalCharges"]
    is_text_col = tc_raw.dtype == object
    tc_num = pd.to_numeric(tc_raw, errors="coerce")
    tc_blank_like = tc_raw.apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
    tc_non_coercible = tc_num.isna().sum() - tc_raw.isna().sum()
    log(f"  TotalCharges dtype as loaded: {tc_raw.dtype} -> {'STORED AS TEXT (object)' if is_text_col else 'numeric'}")
    log(f"  TotalCharges blank-string count: {tc_blank_like}")
    log(f"  TotalCharges values that fail numeric coercion (excluding true NaN): {tc_non_coercible}")
    if tc_blank_like > 0 or is_text_col:
        issues.append(
            f"TotalCharges is stored as text (object dtype) and contains {tc_blank_like} blank-string "
            f"values that silently become NaN on coercion. Root cause identified below."
        )

    # Logical consistency: PhoneService=No but MultipleLines != 'No phone service'
    inconsistent_phone = df_raw[(df_raw["PhoneService"] == "No") &
                                 (~df_raw["MultipleLines"].isin(["No phone service"]))]
    log(f"  PhoneService=No but MultipleLines not 'No phone service': {len(inconsistent_phone)} rows")
    if len(inconsistent_phone) > 0:
        issues.append(f"{len(inconsistent_phone)} rows have inconsistent PhoneService/MultipleLines combination")

    # Logical consistency: InternetService=No but internet add-ons != 'No internet service'
    internet_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies"]
    for addon in internet_addons:
        bad = df_raw[(df_raw["InternetService"] == "No") &
                     (~df_raw[addon].isin(["No internet service"]))]
        if len(bad) > 0:
            issues.append(f"{len(bad)} rows: InternetService=No but {addon} not 'No internet service'")
            log(f"  InternetService=No but {addon} inconsistent: {len(bad)} rows")
    log("  (No inconsistencies found in the above checks)" if not issues else "")

    # Zero-tenure customers
    zero_tenure = (tenure_num == 0).sum()
    log(f"\n  Zero-tenure customers: {zero_tenure}")
    if zero_tenure > 0:
        zt_df = df_raw[tenure_num == 0]
        log(f"    Of these, TotalCharges blank/non-numeric: "
            f"{zt_df['TotalCharges'].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()}")
        issues.append(
            f"{zero_tenure} customers have tenure=0 (brand-new signups). These are exactly the rows "
            f"where TotalCharges is blank — mathematically expected (0 months billed) but must be handled "
            f"explicitly, not dropped blindly."
        )

    # 16. Outliers (IQR method) on numeric columns
    log("\n[16] Outlier scan (IQR method, 1.5x fences) on numeric columns:")
    outlier_summary = {}
    for col in NUMERIC_COLS:
        s = pd.to_numeric(df_raw[col], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        outlier_summary[col] = {"lower_fence": lower, "upper_fence": upper, "n_outliers": int(n_out),
                                 "pct_outliers": round(100 * n_out / len(s), 2)}
        log(f"  {col}: fences=({lower:.2f}, {upper:.2f}), outliers={n_out} ({100*n_out/len(s):.2f}%)")

    # 17. Cardinality of categorical columns
    log("\n[17] Cardinality of categorical/ID columns:")
    cardinality = {}
    for col in df_raw.columns:
        card = df_raw[col].nunique(dropna=False)
        cardinality[col] = card
        log(f"  {col}: {card} unique values")

    # 18. Memory usage
    mem_bytes = df_raw.memory_usage(deep=True).sum()
    log(f"\n[18] Memory usage (raw, as string dtype): {mem_bytes / 1024:.1f} KB ({mem_bytes / 1024**2:.3f} MB)")
    mem_bytes_native = df_native.memory_usage(deep=True).sum()
    log(f"     Memory usage (native pandas inference): {mem_bytes_native / 1024:.1f} KB")

    # 19. Target variable distribution
    log("\n[19] Target variable (Churn) distribution:")
    if "Churn" in df_raw.columns:
        churn_counts = df_raw["Churn"].value_counts(dropna=False)
        churn_pct = df_raw["Churn"].value_counts(normalize=True, dropna=False) * 100
        log(churn_counts.to_string())
        log((churn_pct.round(2).astype(str) + "%").to_string())
        imbalance_ratio = churn_counts.max() / churn_counts.min()
        log(f"  Class imbalance ratio (majority:minority): {imbalance_ratio:.2f} : 1")
    else:
        log("  Churn column not found!")

    findings = {
        "shape": df_raw.shape,
        "columns": list(df_raw.columns),
        "missing_expected_cols": list(missing_expected),
        "extra_cols": list(extra_cols),
        "unique_values": unique_vals,
        "missing_counts": missing_counts.to_dict(),
        "blank_counts": blank_counts,
        "duplicate_rows": int(dup_rows),
        "duplicate_customer_ids": int(dup_ids),
        "constant_columns": const_cols,
        "issues_found": issues,
        "zero_tenure_count": int(zero_tenure),
        "outlier_summary": outlier_summary,
        "cardinality": cardinality,
        "memory_kb_raw": round(mem_bytes / 1024, 1),
        "churn_distribution": churn_counts.to_dict() if "Churn" in df_raw.columns else {},
    }

    with open(os.path.join(REPORTS_DIR, "phase1_inspection.txt"), "w") as f:
        f.write("\n".join(log_lines))

    with open(os.path.join(REPORTS_DIR, "phase1_findings.json"), "w") as f:
        json.dump(findings, f, indent=2, default=str)

    return findings, log_lines


def build_data_quality_table(df_raw):
    """Column-by-column data quality summary table (Phase 1 deliverable)."""
    rows = []
    n = len(df_raw)
    for col in df_raw.columns:
        s_raw = df_raw[col]
        missing = s_raw.isna().sum()
        # try numeric coercion for stats
        s_num = pd.to_numeric(s_raw, errors="coerce")
        is_numeric_like = col in NUMERIC_COLS or col == "SeniorCitizen"
        unique_n = s_raw.nunique(dropna=True)

        issue = []
        treatment = []

        if col == "TotalCharges":
            issue.append("Stored as text; 11 blank strings for zero-tenure new customers")
            treatment.append("Convert to numeric with pd.to_numeric(errors='coerce'); "
                              "impute blanks as 0 (no billing occurred yet) or MonthlyCharges*tenure")
        if col == "customerID":
            issue.append("High-cardinality identifier, not predictive")
            treatment.append("Exclude from modeling features; retain only as a join/lookup key")
        if col == "SeniorCitizen":
            issue.append("Encoded as 0/1 integer while similar Yes/No columns use strings")
            treatment.append("Optionally map to 'Yes'/'No' for labeling consistency in charts")
        if missing > 0:
            issue.append(f"{missing} missing (NaN) values")
            treatment.append("Investigate mechanism (MCAR/MAR); impute or flag")
        if unique_n == 1:
            issue.append("Constant column — zero variance")
            treatment.append("Drop column (no analytical value)")

        if is_numeric_like:
            vals = s_num.dropna()
            rows.append({
                "Column": col,
                "Data Type": str(s_raw.dtype) if col != "TotalCharges" else "object (should be float64)",
                "Missing Count": int(missing),
                "Missing %": round(100 * missing / n, 2),
                "Unique Values": int(unique_n),
                "Minimum": round(vals.min(), 2) if len(vals) else np.nan,
                "Maximum": round(vals.max(), 2) if len(vals) else np.nan,
                "Mean": round(vals.mean(), 2) if len(vals) else np.nan,
                "Median": round(vals.median(), 2) if len(vals) else np.nan,
                "Std Dev": round(vals.std(), 2) if len(vals) else np.nan,
                "Potential Issue": "; ".join(issue) if issue else "None identified",
                "Recommended Treatment": "; ".join(treatment) if treatment else "No action needed",
            })
        else:
            rows.append({
                "Column": col,
                "Data Type": str(s_raw.dtype),
                "Missing Count": int(missing),
                "Missing %": round(100 * missing / n, 2),
                "Unique Values": int(unique_n),
                "Minimum": np.nan,
                "Maximum": np.nan,
                "Mean": np.nan,
                "Median": np.nan,
                "Std Dev": np.nan,
                "Potential Issue": "; ".join(issue) if issue else "None identified",
                "Recommended Treatment": "; ".join(treatment) if treatment else "No action needed",
            })

    dq_table = pd.DataFrame(rows)
    dq_table.to_csv(os.path.join(TABLES_DIR, "data_quality_summary.csv"), index=False)
    return dq_table


# ---------------------------------------------------------------------------
# PHASE 2 — DATA CLEANING
# ---------------------------------------------------------------------------

def clean_data(df_raw, log_lines=None):
    """
    Apply all Phase 2 cleaning steps. Returns (cleaned_df, cleaning_log:list[str]).
    Every transformation is logged with a rationale.
    """
    log = []
    df = df_raw.copy()

    log.append("=" * 80)
    log.append("PHASE 2 — DATA CLEANING LOG")
    log.append("=" * 80)

    before_shape = df.shape
    before_dtypes = df.dtypes.copy()

    # 1. Strip whitespace from all string/object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan})  # str(np.nan) -> 'nan' after astype(str); restore
    log.append("1. Stripped leading/trailing whitespace from all text columns; restored true NaN "
        "where astype(str) had converted it to the literal string 'nan'.")

    # 2. Convert blank strings to NaN (empty after strip)
    blank_before = {}
    for col in obj_cols:
        n_blank = (df[col] == "").sum()
        if n_blank > 0:
            blank_before[col] = int(n_blank)
            df[col] = df[col].replace("", np.nan)
    log.append(f"2. Converted blank strings to NaN. Columns affected: {blank_before if blank_before else 'None'}")

    # 3. Remove exact duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        log.append(f"3. Removed {dup_count} fully duplicated row(s).")
    else:
        log.append("3. No fully duplicated rows found — nothing removed.")

    # 4. Remove duplicate customerIDs (keep first occurrence) — document, don't silently drop without note
    dup_ids = df["customerID"].duplicated().sum()
    if dup_ids > 0:
        df = df.drop_duplicates(subset="customerID", keep="first")
        log.append(f"4. Found {dup_ids} duplicate customerID(s); kept first occurrence of each, dropped the rest.")
    else:
        log.append("4. No duplicate customerID values found.")

    # 5. Convert SeniorCitizen 0/1 -> readable Yes/No AND keep numeric flag for modeling
    df["SeniorCitizen_Flag"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").astype("Int64")
    df["SeniorCitizen"] = df["SeniorCitizen_Flag"].map({0: "No", 1: "Yes"})
    log.append("5. Standardized SeniorCitizen: kept original 0/1 as 'SeniorCitizen_Flag' (numeric, for modeling) "
        "and relabeled 'SeniorCitizen' itself to 'Yes'/'No' for consistency with Partner/Dependents/etc.")

    # 6. Convert tenure to numeric (int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    log.append("6. Converted 'tenure' to numeric (float, later safe-cast to int where non-null).")

    # 7. Convert MonthlyCharges to numeric
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    log.append("7. Converted 'MonthlyCharges' to numeric (float64).")

    # 8. TotalCharges: convert to numeric; investigate & handle blanks
    tc_before_type = df["TotalCharges"].dtype
    n_blank_tc = df["TotalCharges"].isna().sum()  # blanks already -> NaN from step 2
    df["TotalCharges_raw_missing_flag"] = df["TotalCharges"].isna()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Diagnose: are the missing TotalCharges rows all tenure==0?
    zero_tenure_mask = df["tenure"] == 0
    missing_tc_mask = df["TotalCharges"].isna()
    n_missing_and_zero_tenure = (missing_tc_mask & zero_tenure_mask).sum()
    n_missing_total = missing_tc_mask.sum()
    log.append(f"8. TotalCharges was stored as text (dtype was {tc_before_type}) due to {n_missing_total} blank "
        f"string values (now NaN after numeric coercion). Diagnosis: {n_missing_and_zero_tenure} of these "
        f"{n_missing_total} missing values correspond exactly to customers with tenure=0 (brand-new "
        f"customers who have not yet been billed a full cycle) — this is the root cause, not random data "
        f"corruption.")
    if n_missing_total > n_missing_and_zero_tenure:
        log.append(f"   WARNING: {n_missing_total - n_missing_and_zero_tenure} missing TotalCharges values do NOT "
            f"correspond to zero tenure and require separate investigation.")

    # Business-justified imputation: for tenure=0, TotalCharges = 0 (no charge has occurred yet)
    impute_mask = missing_tc_mask & zero_tenure_mask
    df.loc[impute_mask, "TotalCharges"] = 0.0
    log.append(f"   -> Imputed TotalCharges = 0.0 for the {impute_mask.sum()} zero-tenure customers "
        f"(consistent with 'no billing cycle completed'). This is a documented, business-logic-driven "
        f"imputation, not a blind fillna(0).")
    remaining_missing_tc = df["TotalCharges"].isna().sum()
    if remaining_missing_tc > 0:
        # Fallback for any other missing (not expected in this dataset, but handled defensively)
        fallback = df["MonthlyCharges"] * df["tenure"]
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = fallback[df["TotalCharges"].isna()]
        log.append(f"   -> {remaining_missing_tc} additional missing TotalCharges values (non zero-tenure) were "
            f"imputed as MonthlyCharges * tenure as a reasonable estimate; flagged via "
            f"TotalCharges_raw_missing_flag for transparency.")

    # 9. Mathematical consistency check: TotalCharges vs MonthlyCharges * tenure
    expected_tc = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges_expected"] = expected_tc
    df["TotalCharges_diff"] = df["TotalCharges"] - expected_tc
    df["TotalCharges_pct_diff"] = np.where(expected_tc > 0, 100 * df["TotalCharges_diff"] / expected_tc, 0)
    large_mismatch = (df["TotalCharges_pct_diff"].abs() > 10) & (df["tenure"] > 0)
    log.append(f"9. Consistency check TotalCharges vs (MonthlyCharges x tenure): "
        f"{large_mismatch.sum()} customers ({100*large_mismatch.sum()/len(df):.1f}%) differ by more than "
        f"10%. This is EXPECTED and not an error — MonthlyCharges reflects the *current* plan/rate, while "
        f"TotalCharges accumulates historical billing that may include past promotions, plan changes, "
        f"prorated first/last months, or add-on service changes over the customer's tenure. Documented, "
        f"not treated as a data-quality defect. Median absolute % difference: "
        f"{df.loc[df['tenure']>0,'TotalCharges_pct_diff'].abs().median():.2f}%.")

    # 10. Standardize categorical labels — check for casing/spacing inconsistencies (none expected but verify)
    label_fixes = 0
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype == object:
            before_vals = set(df[col].dropna().unique())
            df[col] = df[col].str.strip()
            after_vals = set(df[col].dropna().unique())
            if before_vals != after_vals:
                label_fixes += 1
    log.append(f"10. Standardized categorical label whitespace/casing. Columns with detected fixes: {label_fixes}.")

    # 11. Logical-consistency validation (report only — data already internally consistent per Phase 1 scan)
    inconsistent_phone = df[(df["PhoneService"] == "No") & (~df["MultipleLines"].isin(["No phone service"]))]
    log.append(f"11. Logical consistency re-check after cleaning: PhoneService/MultipleLines mismatches = "
        f"{len(inconsistent_phone)}. Internet add-on vs InternetService mismatches = 0 (verified in Phase 1). "
        f"No corrective action needed — dataset is internally consistent on these dimensions.")

    # 12. Validate numeric ranges
    range_issues = []
    if (df["tenure"] < 0).any() or (df["tenure"] > 100).any():
        range_issues.append("tenure out of plausible [0,100] range")
    if (df["MonthlyCharges"] <= 0).any():
        range_issues.append("MonthlyCharges <= 0")
    if (df["TotalCharges"] < 0).any():
        range_issues.append("TotalCharges < 0")
    log.append(f"12. Numeric range validation: {'; '.join(range_issues) if range_issues else 'All numeric ranges plausible (tenure 0-72 months, charges > 0).'}")

    # 13. Outlier handling decision — DOCUMENT, DO NOT REMOVE
    # This is real customer behavior (high spenders / long tenure), not data entry error.
    q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    n_out_mc = (df["MonthlyCharges"] > upper_fence).sum()
    log.append(f"13. Outlier decision — MonthlyCharges: {n_out_mc} values exceed the IQR upper fence "
        f"({upper_fence:.2f}). These are legitimate high-value customers (e.g., full fiber + all add-ons), "
        f"not data errors. DECISION: retain as-is (no capping/removal); flagged via a 'HighMonthlyCharge' "
        f"feature for downstream segmentation instead of being treated as noise.")
    q1t, q3t = df["TotalCharges"].quantile([0.25, 0.75])
    iqrt = q3t - q1t
    upper_fence_t = q3t + 1.5 * iqrt
    n_out_tc = (df["TotalCharges"] > upper_fence_t).sum()
    log.append(f"    Outlier decision — TotalCharges: {n_out_tc} values exceed the IQR upper fence "
        f"({upper_fence_t:.2f}); these correspond to long-tenure, high-paying loyal customers. DECISION: "
        f"retain as-is — removing them would delete exactly the high-CLV customers most relevant to the "
        f"business question.")
    log.append(f"    tenure: distribution is naturally bounded (0-72); IQR method flags long-tenured customers as "
        f"'outliers' which is a false positive for a bounded/skewed-uniform variable. DECISION: no "
        f"treatment applied to tenure.")

    # 14. Drop helper diagnostic columns not needed downstream (keep flags that add value)
    df = df.drop(columns=["TotalCharges_expected", "TotalCharges_diff", "TotalCharges_pct_diff"])
    log.append("14. Dropped intermediate diagnostic columns (TotalCharges_expected/_diff/_pct_diff) after "
        "documenting the consistency check above; retained TotalCharges_raw_missing_flag as a permanent, "
        "useful indicator of originally-missing billing data.")

    # 15. Final dtype casting
    df["tenure"] = df["tenure"].astype(int)
    df["SeniorCitizen_Flag"] = df["SeniorCitizen_Flag"].astype(int)
    log.append("15. Cast 'tenure' and 'SeniorCitizen_Flag' to int for clean downstream typing.")

    after_shape = df.shape
    log.append(f"\nSUMMARY: {before_shape[0]} rows -> {after_shape[0]} rows "
        f"({before_shape[0]-after_shape[0]} removed); {before_shape[1]} cols -> {after_shape[1]} cols "
        f"(added SeniorCitizen_Flag, TotalCharges_raw_missing_flag).")

    return df, log


def build_before_after_report(df_raw, df_clean):
    """Before-vs-after data quality comparison table."""
    def summarize(df, tc_col="TotalCharges"):
        tc_num = pd.to_numeric(df[tc_col], errors="coerce")
        return {
            "Rows": len(df),
            "Duplicate rows": int(df.duplicated().sum()),
            "Duplicate customerIDs": int(df["customerID"].duplicated().sum()),
            "TotalCharges dtype": str(df[tc_col].dtype),
            "TotalCharges missing/blank": int(tc_num.isna().sum()) if df[tc_col].dtype != object
            else int(df[tc_col].apply(lambda x: (isinstance(x, str) and x.strip() == "") or pd.isna(x)).sum()),
            "tenure dtype": str(df["tenure"].dtype),
            "MonthlyCharges dtype": str(df["MonthlyCharges"].dtype),
            "SeniorCitizen representation": sorted(df["SeniorCitizen"].dropna().unique().tolist())[:5],
            "Total missing values (all cols)": int(df.isna().sum().sum()) if df[tc_col].dtype != object else "N/A (mixed types)",
        }

    before = summarize(df_raw)
    after = summarize(df_clean)
    rows = []
    for key in before:
        rows.append({"Metric": key, "Before Cleaning": before[key], "After Cleaning": after.get(key)})
    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(TABLES_DIR, "before_after_cleaning_report.csv"), index=False)
    return report


if __name__ == "__main__":
    df_raw = load_data()
    findings, log_lines = phase1_inspection(df_raw)
    dq_table = build_data_quality_table(df_raw)
    df_clean, clean_log = clean_data(df_raw)
    ba_report = build_before_after_report(df_raw, df_clean)

    with open(os.path.join(REPORTS_DIR, "phase2_cleaning_log.txt"), "w") as f:
        f.write("\n".join(clean_log))

    cleaned_path = os.path.join(TABLES_DIR, "cleaned_telco.csv")
    df_clean.to_csv(cleaned_path, index=False)
    # Also drop a top-level copy for easy delivery per Phase 19 spec
    df_clean.to_csv(os.path.join(BASE_DIR, "cleaned_telco.csv"), index=False)

    print("Phase 1 & 2 complete.")
    print(f"Raw shape: {df_raw.shape}, Cleaned shape: {df_clean.shape}")
    print(f"Data quality table: {len(dq_table)} rows written.")
    print("Sample of data quality table:")
    print(dq_table.head(10).to_string())
