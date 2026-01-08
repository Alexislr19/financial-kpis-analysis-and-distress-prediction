"""
Entry point - THIS MUST WORK.

Reproduces notebook 04 (modeling):
- load final panel
- build X and y
- train/test split + median imputation
- train Logistic Regression, Random Forest, XGBoost
- print metrics & classification reports
- save confusion matrices, ROC curves, XGBoost importances in results/
"""

import warnings
warnings.filterwarnings("ignore")

from src.data_loader import (
    load_final_panel,
    build_X_y_from_panel,
    standard_train_test_split,
    impute_train_test_median,
)
from src.models import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost_classifier,
    create_standard_scaler,
)
from src.evaluation import (
    evaluate_single_model,
    save_roc_curves_multi,
    print_final_summary,
    print_detailed_reports,
)


def main() -> None:
    # =========================
    # 1. Load final panel
    # =========================
    df = load_final_panel()  # lit data/final/panel_balanced_with_deltas.csv
    print("Final panel shape:", df.shape)

    # =========================
    # 2. Build X and y (same columns as notebook 04)
    # =========================
    X, y, feature_names = build_X_y_from_panel(df)

    print("\nClass distribution in y:")
    print(y.value_counts())

    print("\nX shape:", X.shape)
    print("Any NaN in X ? ->", X.isna().any().any())

    # =========================
    # 3. Train / test split + imputation
    # =========================
    X_train, X_test, y_train, y_test = standard_train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=True
    )

    # median imputation (SimpleImputer)
    X_train_imp, X_test_imp, _ = impute_train_test_median(X_train, X_test)

    # =========================
    # 4. Train models (same logic as notebook 04)
    # =========================
    results = {}          # accuracy + AUC (comme notebook)
    probas_test = {}      # pour les ROC curves multipanel
    preds_test = {}       # pour les rapports détaillés

    # ----- Logistic Regression -----
    scaler = create_standard_scaler()
    X_train_lr = scaler.fit_transform(X_train_imp)
    X_test_lr = scaler.transform(X_test_imp)

    lr = create_logistic_regression(max_iter=1000)
    lr.fit(X_train_lr, y_train)

    y_pred_lr = lr.predict(X_test_lr)
    y_proba_lr = lr.predict_proba(X_test_lr)[:, 1]

    preds_test["Logistic"] = y_pred_lr
    probas_test["Logistic"] = y_proba_lr

    # On garde accuracy + AUC comme dans le dict "results" du notebook
    from sklearn.metrics import accuracy_score, roc_auc_score
    results["Logistic"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "auc": roc_auc_score(y_test, y_proba_lr),
    }

    # Evaluation + matrices + métriques complètes (Accuracy, Precision, Recall, F1, AUC)
    metrics_log = evaluate_single_model(
        y_true=y_test,
        y_pred=y_pred_lr,
        y_proba=y_proba_lr,
        model_name="Logistic Regression",
    )

    # ----- Random Forest -----
    rf = create_random_forest()
    rf.fit(X_train_imp, y_train)

    y_pred_rf = rf.predict(X_test_imp)
    y_proba_rf = rf.predict_proba(X_test_imp)[:, 1]

    preds_test["RandomForest"] = y_pred_rf
    probas_test["RandomForest"] = y_proba_rf

    results["RandomForest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "auc": roc_auc_score(y_test, y_proba_rf),
    }

    metrics_rf = evaluate_single_model(
        y_true=y_test,
        y_pred=y_pred_rf,
        y_proba=y_proba_rf,
        model_name="Random Forest",
        feature_importances=rf.feature_importances_,
        feature_names=feature_names,
    )

    # ----- XGBoost -----
    xgb_model = create_xgboost_classifier()
    xgb_model.fit(X_train_imp, y_train)

    y_pred_xgb = xgb_model.predict(X_test_imp)
    y_proba_xgb = xgb_model.predict_proba(X_test_imp)[:, 1]

    preds_test["XGBoost"] = y_pred_xgb
    probas_test["XGBoost"] = y_proba_xgb

    results["XGBoost"] = {
        "accuracy": accuracy_score(y_test, y_pred_xgb),
        "auc": roc_auc_score(y_test, y_proba_xgb),
    }

    metrics_xgb = evaluate_single_model(
        y_true=y_test,
        y_pred=y_pred_xgb,
        y_proba=y_proba_xgb,
        model_name="XGBoost",
        feature_importances=xgb_model.feature_importances_,
        feature_names=feature_names,
    )

    # =========================
    # 5. ROC curves multi-modèles (section 6)
    # =========================
    # On construit un dict AUC pour chaque modèle (comme dans le notebook)
    auc_dict = {
        name: res["auc"] for name, res in results.items()
    }
    save_roc_curves_multi(y_test, probas_test, auc_dict)

    # =========================
    # 6. Final metrics table (section 7)
    # =========================
    # On prend les métriques complètes retournées par evaluate_single_model
    metrics_all = {
        "Logistic": metrics_log,
        "RandomForest": metrics_rf,
        "XGBoost": metrics_xgb,
    }
    print_final_summary(metrics_all)

    # =========================
    # 7. Detailed reports per model (section 9)
    # =========================
    print_detailed_reports(y_test, preds_test)


if __name__ == "__main__":
    main()

