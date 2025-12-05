"""
Evaluation and visualization utilities for classification models.
All plots are automatically saved into the /results folder.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

sns.set(style="whitegrid")

# -----------------------------------------------------------------------------
#  Directories for results
# -----------------------------------------------------------------------------
RESULTS_DIR = "results"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROC_DIR = os.path.join(RESULTS_DIR, "roc_curves")
IMP_DIR = os.path.join(RESULTS_DIR, "importances")

for d in [RESULTS_DIR, CM_DIR, ROC_DIR, IMP_DIR]:
    os.makedirs(d, exist_ok=True)


# -----------------------------------------------------------------------------
#  Basic helpers
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Same metrics as in notebook 04."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (1)": precision_score(y_true, y_pred, pos_label=1),
        "Recall (1)": recall_score(y_true, y_pred, pos_label=1),
        "F1-score (1)": f1_score(y_true, y_pred, pos_label=1),
        "AUC": roc_auc_score(y_true, y_proba),
    }


def report_to_df(y_true, y_pred) -> pd.DataFrame:
    """DataFrame version of classification_report (comme notebook 04)."""
    rep = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(rep).T.round(3)


def save_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Save confusion matrix as PNG in results/confusion_matrices/."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        annot_kws={"size": 14, "weight": "bold", "color": "black"},
    )
    plt.title(f"{model_name} – Confusion Matrix", fontsize=14, pad=10)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    out_path = os.path.join(CM_DIR, f"{model_name}_cm.png")
    plt.savefig(out_path)
    plt.close()


def save_roc_curves_multi(
    y_true,
    probas_dict: Dict[str, np.ndarray],
    auc_dict: Dict[str, float],
    filename: str = "roc_all_models.png",
) -> None:
    """
    Save ROC curves for all models on the same figure.
    Correspond à la section 6 du notebook.
    """
    plt.figure(figsize=(6, 6))

    for name, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = auc_dict[name]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(ROC_DIR, filename)
    plt.savefig(out_path)
    plt.close()


def save_xgb_importances(importances: np.ndarray, feature_names, top_n: int = 15) -> None:
    """
    Save barplot of top XGBoost feature importances.
    Correspond à la section 8 du notebook.
    """
    feat_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=feat_imp.head(top_n), x="importance", y="feature")
    plt.title("XGBoost - Top 15 feature importances")
    plt.tight_layout()
    out_path = os.path.join(IMP_DIR, "XGBoost_importances.png")
    plt.savefig(out_path)
    plt.close()

    # affichage texte
    print("\nTop 15 XGBoost feature importances:")
    print(feat_imp.head(top_n))


# -----------------------------------------------------------------------------
#  Wrapper par modèle (métriques + matrices de confusion)
# -----------------------------------------------------------------------------
def evaluate_single_model(
    y_true,
    y_pred,
    y_proba,
    model_name: str,
    feature_importances=None,
    feature_names=None,
) -> Dict[str, float]:
    """
    Reproduit le comportement du notebook :
    - print du classification_report
    - calcul des métriques (Accuracy, Precision, Recall, F1, AUC)
    - sauvegarde de la matrice de confusion
    - optionnel: sauvegarde des importances
    """
    print(f"\n===== {model_name} =====")
    print(classification_report(y_true, y_pred, digits=3))

    # metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # confusion matrix
    save_confusion_matrix(y_true, y_pred, model_name)

    # feature importances (seulement RF / XGB)
    if feature_importances is not None and feature_names is not None:
        save_xgb_importances(feature_importances, feature_names)

    return metrics


def print_final_summary(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Construit et affiche le tableau final des métriques
    (équivalent à summary_clean dans le notebook).
    """
    summary_df = pd.DataFrame(metrics_dict).T
    print("\n===== Final Model Comparison =====")
    print(summary_df.round(3))


def print_detailed_reports(y_test, preds_dict: Dict[str, np.ndarray]) -> None:
    """
    Reproduit la section 9 du notebook :
    tables détaillées (classification_report en DataFrame).
    """
    for name, y_pred in preds_dict.items():
        df_rep = report_to_df(y_test, y_pred)
        print(f"\n===== {name} – detailed report =====")
        print(df_rep.to_string())




