# Telco Customer Churn — Complete Analytics Project

A full beginner-to-advanced analysis of a telecom customer churn dataset
(7,043 customers, 21 original fields), covering data quality auditing,
cleaning, exploratory analysis, churn/tenure/revenue/service/contract
deep-dives, statistical significance testing, customer segmentation,
churn-driver ranking, machine-learning churn prediction, and a management-
ready HTML report with 19 prioritized business insights and phased
recommendations.

## Headline Results

- **Overall churn rate: 26.54%** (1,869 of 7,043 customers)
- **Estimated annualized revenue at risk: ~$1.67M** (30.5% of total MRR)
- **Top churn drivers (combined statistical + ML ranking):** Contract type,
  tenure, TotalCharges, OnlineSecurity, InternetService, TechSupport,
  PaymentMethod
- **Best model by ROC-AUC:** Gradient Boosting (0.847); **recommended for
  deployment:** Random Forest (0.845 ROC-AUC, 77.3% recall on the churn
  class — far more useful for a "catch the churners" business objective)
- **Highest-risk profile:** Month-to-month + Fiber optic + Electronic check
  (60.4% churn rate, ~1,300 customers)

## Project Structure

```
telco_analysis/
├── data/
│   └── telco.csv                  # Original, untouched raw dataset
├── outputs/
│   ├── figures/                   # 13 presentation-ready PNG charts
│   ├── tables/                    # ~30 CSV analysis tables
│   └── reports/                   # Text logs + the final HTML report
├── src/
│   ├── data_cleaning.py           # Phase 1-2: inspection & cleaning
│   ├── features.py                # Phase 11: feature engineering
│   ├── eda.py                     # Phase 3: univariate analysis
│   ├── churn_analysis.py          # Phase 4-8: churn/tenure/revenue/service/contract
│   ├── statistical_analysis.py    # Phase 9: significance testing
│   ├── segmentation.py            # Phase 10: customer segments
│   ├── modeling.py                # Phase 12-14: ML models & driver ranking
│   └── reporting.py               # Phase 15-19: insights, exec summary, HTML report
├── main.py                        # Runs the entire pipeline end-to-end
├── cleaned_telco.csv              # Deliverable: cleaned dataset
├── churn_analysis_summary.csv     # Deliverable: consolidated churn metrics
├── key_insights.csv               # Deliverable: 19 structured business insights
├── analysis_report.html           # Deliverable: final management report
├── requirements.txt
└── README.md
```

## How to Reproduce

```bash
pip install -r requirements.txt
python3 main.py
```

This regenerates every table, figure, and the final HTML report from the
raw `data/telco.csv` in one run (typically well under a minute). The
original `data/telco.csv` is never modified — all cleaning happens on an
in-memory copy, and cleaned output is written to a new file.

`random_state=42` is used everywhere randomness is involved (train/test
split, all models, cross-validation) for full reproducibility.

## Key Deliverables

| File | Description |
|---|---|
| `analysis_report.html` | Full management-ready report: executive summary, all analysis phases, 19 prioritized insights, recommendations |
| `cleaned_telco.csv` | Cleaned dataset (TotalCharges fixed, SeniorCitizen standardized, diagnostic flags added) |
| `churn_analysis_summary.csv` | Consolidated churn-rate metrics across all key cuts |
| `customer_segment_summary.csv` | 7 business segments ranked by revenue exposure |
| `model_comparison.csv` | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost compared on Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC |
| `key_insights.csv` | 19 insights in Finding/Evidence/Why-it-matters/Action/Impact/Confidence format |
| `outputs/tables/final_ranked_churn_drivers.csv` | Combined statistical + ML-based churn driver ranking |

## Methodology Notes

- **Data quality:** `TotalCharges` was found stored as text due to 11 blank
  values, all exactly matching `tenure=0` (brand-new customers not yet
  billed) — imputed as 0.0 with the reasoning documented in
  `outputs/reports/phase2_cleaning_log.txt`. No duplicate rows, no duplicate
  customer IDs, no logical data inconsistencies were found.
- **Statistics:** Chi-square + Cramér's V for categorical associations,
  point-biserial correlation for numeric-vs-churn relationships, Pearson/
  Spearman for numeric-numeric relationships. Effect size (not just
  p-value) drives prioritization, since large-sample significance can be
  business-trivial.
- **Modeling:** Stratified 75/25 train/test split; 5-fold stratified
  cross-validation; `customerID` excluded from features (identifier
  leakage); class imbalance handled via `class_weight="balanced"` /
  `scale_pos_weight`. Recall on the churn class is treated as the primary
  business-usefulness metric alongside ROC-AUC, not accuracy.
- **Causality:** All drivers are explicitly labeled as correlational/
  associational or predictive-importance — this analysis does not claim
  causation anywhere; recommendations note where A/B validation is needed
  before scaling spend.

## Limitations

See the "Limitations" section of `analysis_report.html` for the full list —
in short: this is single-snapshot, observational data with no service-
quality, competitor, or macroeconomic fields, so several open questions
(especially *why* Fiber optic churns so much higher than DSL) require
additional data collection to resolve conclusively.
