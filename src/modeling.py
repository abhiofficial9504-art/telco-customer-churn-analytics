"""
modeling.py
===========
Phase 12 — Churn Driver Analysis (statistical + model-based, combined)
Phase 13 — Machine Learning (Logistic Regression, Decision Tree, Random
           Forest, Gradient Boosting / XGBoost)
Phase 14 — Model Interpretation (feature importance, permutation importance,
           business translation of predictions)

Design notes / anti-leakage rules followed:
- customerID is excluded from all features.
- TotalCharges is a function of tenure x MonthlyCharges (near-collinear, not
  a leak per se — it's known at prediction time for an existing customer —
  but we flag the collinearity and still include it since a live scoring
  system would have this value for any current customer).
- Engineered features (TenureGroup, NumberOfServices, ContractRisk,
  PaymentRisk, HighValueCustomer, etc.) are all derived strictly from
  columns available at prediction time (no post-outcome information, no
  use of Churn itself in their construction) -> no leakage.
- Train/test split is stratified on the target; random_state=42 everywhere.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve,
                              precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

RANDOM_STATE = 42

# Features used for modeling (deliberately excludes customerID; excludes the
# raw diagnostic flag TotalCharges_raw_missing_flag as it is a data-quality
# artifact, not a business driver; excludes CLV_realized as it's a simple
# alias of TotalCharges = would be redundant/near-duplicate leakage of the
# same info already in TotalCharges).
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumberOfServices",
                     "ServiceBundleSize", "AverageChargePerMonth"]


def prepare_xy(df):
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ])


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                                    class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                                 random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                                 random_state=RANDOM_STATE, class_weight="balanced",
                                                 n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                          learning_rate=0.05, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
            eval_metric="logloss", scale_pos_weight=(1869 and None) or 1,  # set below dynamically
            use_label_encoder=False,
        )
    return models


def run_modeling(df):
    X, y = prepare_xy(df)
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor()
    models = get_models()
    if HAS_XGB:
        # set scale_pos_weight properly using train split class ratio
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        models["XGBoost"].set_params(scale_pos_weight=neg / pos)

    results = []
    fitted_pipelines = {}
    roc_data = {}
    pr_data = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Churn class)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "CV ROC-AUC (mean +/- std, 5-fold)": f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc)
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = (precision_arr, recall_arr, pr_auc)

    comparison = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    comparison.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)

    best_model_name = comparison.iloc[0]["Model"]
    best_pipe = fitted_pipelines[best_model_name]

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "comparison": comparison, "fitted_pipelines": fitted_pipelines,
        "roc_data": roc_data, "pr_data": pr_data,
        "best_model_name": best_model_name, "best_pipe": best_pipe,
    }


def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Prediction Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "15_roc_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(best_pipe, X_test, y_test, best_model_name):
    y_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted: No Churn", "Predicted: Churn"],
                yticklabels=["Actual: No Churn", "Actual: Churn"])
    ax.set_title(f"Confusion Matrix — {best_model_name} (Test Set)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "14_confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path, cm


def get_feature_names(preprocessor):
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


def feature_importance_analysis(fitted_pipelines, X_test, y_test):
    """Random Forest impurity-based importance + permutation importance
    (model-agnostic, computed on the held-out test set to avoid overfitting
    bias) + Logistic Regression coefficients (direction of effect)."""
    rf_pipe = fitted_pipelines["Random Forest"]
    feature_names = get_feature_names(rf_pipe.named_steps["prep"])

    rf_importances = rf_pipe.named_steps["clf"].feature_importances_
    rf_imp_df = pd.DataFrame({"Feature": feature_names, "RF_Importance": rf_importances})
    rf_imp_df = rf_imp_df.sort_values("RF_Importance", ascending=False)

    # Permutation importance (model-agnostic, on test set)
    perm = permutation_importance(rf_pipe, X_test, y_test, n_repeats=15,
                                   random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
    }).sort_values("Permutation_Importance_Mean", ascending=False)

    # Logistic regression coefficients (direction + magnitude, standardized features)
    lr_pipe = fitted_pipelines["Logistic Regression"]
    lr_feature_names = get_feature_names(lr_pipe.named_steps["prep"])
    lr_coefs = lr_pipe.named_steps["clf"].coef_[0]
    lr_df = pd.DataFrame({"Feature": lr_feature_names, "LogReg_Coefficient": lr_coefs})
    lr_df["Direction"] = np.where(lr_df["LogReg_Coefficient"] > 0, "Increases Churn Risk",
                                   "Decreases Churn Risk (Retention)")
    lr_df = lr_df.reindex(lr_df["LogReg_Coefficient"].abs().sort_values(ascending=False).index)

    rf_imp_df.to_csv(os.path.join(TABLES_DIR, "feature_importance_random_forest.csv"), index=False)
    perm_df.to_csv(os.path.join(TABLES_DIR, "feature_importance_permutation.csv"), index=False)
    lr_df.to_csv(os.path.join(TABLES_DIR, "feature_importance_logistic_regression.csv"), index=False)

    return rf_imp_df, perm_df, lr_df


def plot_feature_importance(rf_imp_df, perm_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    top_rf = rf_imp_df.head(15).sort_values("RF_Importance")
    axes[0].barh(top_rf["Feature"], top_rf["RF_Importance"], color="#4C72B0")
    axes[0].set_title("Top 15 Features — Random Forest Importance")
    axes[0].set_xlabel("Importance")

    top_perm = perm_df.head(15).sort_values("Permutation_Importance_Mean")
    axes[1].barh(top_perm["Feature"], top_perm["Permutation_Importance_Mean"],
                 xerr=top_perm["Permutation_Importance_Std"], color="#DD8452")
    axes[1].set_title("Top 15 Features — Permutation Importance (ROC-AUC drop)")
    axes[1].set_xlabel("Mean Importance (AUC decrease)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "13_churn_driver_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def build_ranked_churn_driver_table(sig_table, rf_imp_df, perm_df, lr_df):
    """Combine statistical significance + model importance into one final
    ranked churn-driver table with clear correlation-vs-prediction language."""
    # Aggregate one-hot importance back to the parent feature for readability
    def parent_feature(f):
        for cat in ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"]:
            if f.startswith(cat):
                return cat
        return f

    rf_agg = rf_imp_df.copy()
    rf_agg["Parent"] = rf_agg["Feature"].apply(parent_feature)
    rf_agg = rf_agg.groupby("Parent")["RF_Importance"].sum().reset_index()
    rf_agg = rf_agg.rename(columns={"Parent": "Feature"})

    perm_agg = perm_df.copy()
    perm_agg["Parent"] = perm_agg["Feature"].apply(parent_feature)
    perm_agg = perm_agg.groupby("Parent")["Permutation_Importance_Mean"].sum().reset_index()
    perm_agg = perm_agg.rename(columns={"Parent": "Feature"})

    sig_small = sig_table[["Feature", "Effect Size Label", "Significant (alpha=0.05)"]].copy()

    merged = sig_small.merge(rf_agg, on="Feature", how="left").merge(perm_agg, on="Feature", how="left")
    merged["RF_Importance"] = merged["RF_Importance"].fillna(0)
    merged["Permutation_Importance_Mean"] = merged["Permutation_Importance_Mean"].fillna(0)
    merged["Combined_Rank_Score"] = (
        merged["RF_Importance"].rank(pct=True) * 0.5 +
        merged["Permutation_Importance_Mean"].rank(pct=True) * 0.5
    )
    merged = merged.sort_values("Combined_Rank_Score", ascending=False).reset_index(drop=True)
    merged.insert(0, "Rank", range(1, len(merged) + 1))
    merged.to_csv(os.path.join(TABLES_DIR, "final_ranked_churn_drivers.csv"), index=False)
    return merged


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(TABLES_DIR, "telco_features.csv"))
    modeling_results = run_modeling(df)
    print("\nModel comparison:")
    print(modeling_results["comparison"].to_string())
    print(f"\nBest model by ROC-AUC: {modeling_results['best_model_name']}")

    plot_roc_curves(modeling_results["roc_data"])
    path, cm = plot_confusion_matrix(modeling_results["best_pipe"], modeling_results["X_test"],
                                      modeling_results["y_test"], modeling_results["best_model_name"])
    print("\nConfusion matrix (best model):")
    print(cm)

    rf_imp_df, perm_df, lr_df = feature_importance_analysis(
        modeling_results["fitted_pipelines"], modeling_results["X_test"], modeling_results["y_test"]
    )
    print("\nTop 10 Random Forest feature importances:")
    print(rf_imp_df.head(10).to_string())
    print("\nTop 10 permutation importances:")
    print(perm_df.head(10).to_string())
    print("\nTop 10 logistic regression coefficients (by magnitude):")
    print(lr_df.head(10).to_string())

    plot_feature_importance(rf_imp_df, perm_df)

    sig_table = pd.read_csv(os.path.join(TABLES_DIR, "statistical_significance_table.csv"))
    driver_table = build_ranked_churn_driver_table(sig_table, rf_imp_df, perm_df, lr_df)
    print("\nFinal ranked churn driver table:")
    print(driver_table.to_string())
