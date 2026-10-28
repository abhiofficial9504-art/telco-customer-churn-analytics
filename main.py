"""
main.py
=======
Single entry point that runs the complete Telco Customer Churn analysis
pipeline end-to-end, in order:

  Phase 1-2  : src/data_cleaning.py   -> inspection + cleaning
  Phase 11   : src/features.py        -> feature engineering
  Phase 3    : src/eda.py             -> univariate analysis
  Phase 4-8  : src/churn_analysis.py  -> churn/tenure/revenue/service/contract
  Phase 9    : src/statistical_analysis.py -> significance testing
  Phase 10   : src/segmentation.py    -> customer segments
  Phase 12-14: src/modeling.py        -> ML models + driver ranking
  Phase 15-19: src/reporting.py       -> insights, exec summary, HTML report

Run with:  python3 main.py
All outputs are written to outputs/{figures,tables,reports}/ and the
top-level deliverable files (cleaned_telco.csv, key_insights.csv,
churn_analysis_summary.csv, analysis_report.html, etc.) are also copied to
the project root for easy delivery.

random_state=42 is used everywhere randomness is involved, per project
working rules. The original data/telco.csv is never modified.
"""
import sys
import time

sys.path.insert(0, "src")

import data_cleaning
import features
import eda
import churn_analysis
import statistical_analysis
import segmentation
import modeling
import reporting


def run_pipeline():
    t0 = time.time()
    print("=" * 80)
    print("TELCO CUSTOMER CHURN — FULL ANALYSIS PIPELINE")
    print("=" * 80)

    # Phase 1 & 2
    print("\n[1/8] Phase 1-2: Data inspection & cleaning...")
    df_raw = data_cleaning.load_data()
    findings, _ = data_cleaning.phase1_inspection(df_raw)
    data_cleaning.build_data_quality_table(df_raw)
    df_clean, clean_log = data_cleaning.clean_data(df_raw)
    data_cleaning.build_before_after_report(df_raw, df_clean)
    import os
    with open(os.path.join(data_cleaning.REPORTS_DIR, "phase2_cleaning_log.txt"), "w") as f:
        f.write("\n".join(clean_log))
    df_clean.to_csv(os.path.join(data_cleaning.TABLES_DIR, "cleaned_telco.csv"), index=False)
    df_clean.to_csv(os.path.join(data_cleaning.BASE_DIR, "cleaned_telco.csv"), index=False)
    print(f"    Done. Raw shape={df_raw.shape}, Cleaned shape={df_clean.shape}")

    # Phase 11 (feature engineering happens before other analyses so all
    # downstream phases can use engineered features)
    print("\n[2/8] Phase 11: Feature engineering...")
    df = features.engineer_features(df_clean)
    df.to_csv(os.path.join(features.TABLES_DIR, "telco_features.csv"), index=False)
    doc_df = features.get_feature_documentation()
    import pandas as pd
    pd.DataFrame([{"Feature": k, "Description": v} for k, v in doc_df.items()]).to_csv(
        os.path.join(features.TABLES_DIR, "feature_documentation.csv"), index=False)
    print(f"    Done. Shape with engineered features: {df.shape}")

    # Phase 3
    print("\n[3/8] Phase 3: Univariate EDA...")
    eda.run_eda(df)
    print("    Done. Numeric/categorical summaries and distribution charts written.")

    # Phase 4-8
    print("\n[4/8] Phase 4-8: Churn, tenure, revenue, service, contract/payment analysis...")
    kpis = churn_analysis.overall_churn_kpis(df)
    combined_cat = churn_analysis.churn_by_all_categoricals(df)
    churn_analysis.identify_strongest_risk_groups(combined_cat)
    churn_analysis.churn_by_numeric_bins(df)
    tenure_tab = churn_analysis.tenure_band_analysis(df)
    churn_analysis.plot_tenure_churn(tenure_tab)
    churn_analysis.revenue_analysis(df)
    churn_analysis.plot_revenue_at_risk(df)
    service_tab, nos_grp = churn_analysis.service_adoption_analysis(df)
    churn_analysis.plot_service_adoption(service_tab, nos_grp)
    contract_tab, payment_tab, paperless_tab, combo_tab = churn_analysis.contract_payment_analysis(df)
    churn_analysis.plot_contract_payment(contract_tab, payment_tab, combo_tab)
    churn_analysis.plot_churn_kpi_and_distribution(df, kpis)
    print(f"    Done. Overall churn rate: {kpis['Churn Rate (%)']}%")

    # Phase 9
    print("\n[5/8] Phase 9: Statistical significance testing...")
    pearson, spearman = statistical_analysis.numeric_correlation_matrix(df)
    statistical_analysis.plot_correlation_heatmap(pearson, spearman)
    statistical_analysis.significance_table(df)
    statistical_analysis.numeric_numeric_significance(df)
    statistical_analysis.crosstab_pivot_tables(df)
    print("    Done. Significance table + correlation heatmaps written.")

    # Phase 10
    print("\n[6/8] Phase 10: Customer segmentation...")
    seg_table = segmentation.segment_summary(df)
    segmentation.plot_segments(seg_table)
    print(f"    Done. {len(seg_table)} business segments defined.")

    # Phase 12-14
    print("\n[7/8] Phase 12-14: Machine learning models & churn driver ranking...")
    modeling_results = modeling.run_modeling(df)
    modeling.plot_roc_curves(modeling_results["roc_data"])
    modeling.plot_confusion_matrix(modeling_results["best_pipe"], modeling_results["X_test"],
                                    modeling_results["y_test"], modeling_results["best_model_name"])
    rf_imp_df, perm_df, lr_df = modeling.feature_importance_analysis(
        modeling_results["fitted_pipelines"], modeling_results["X_test"], modeling_results["y_test"]
    )
    modeling.plot_feature_importance(rf_imp_df, perm_df)
    sig_table = pd.read_csv(os.path.join(modeling.TABLES_DIR, "statistical_significance_table.csv"))
    modeling.build_ranked_churn_driver_table(sig_table, rf_imp_df, perm_df, lr_df)
    print(f"    Done. Best model by ROC-AUC: {modeling_results['best_model_name']} "
          f"({modeling_results['comparison'].iloc[0]['ROC-AUC']})")

    # Phase 15-19
    print("\n[8/8] Phase 15-19: Insights, executive summary, recommendations, final report...")
    insights = reporting.build_key_insights()
    reporting.key_insights_to_df(insights)
    reporting.build_churn_analysis_summary()
    reporting.build_executive_summary()
    reporting.build_recommendations()
    report_path = reporting.build_html_report()
    print(f"    Done. Final report: {report_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
