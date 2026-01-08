"""
Model definitions for bankruptcy prediction:
- Logistic Regression
- Random Forest
- XGBoost
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ===== Logistic Regression =====

def create_logistic_regression(max_iter: int = 1000) -> LogisticRegression:
    """Return a LogisticRegression model instance."""
    return LogisticRegression(max_iter=max_iter)


def create_standard_scaler() -> StandardScaler:
    """Return a StandardScaler instance."""
    return StandardScaler()


# ===== Random Forest =====

def create_random_forest() -> RandomForestClassifier:
    """Return a RandomForestClassifier with default hyperparameters used in notebook 04."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )


# ===== XGBoost =====

def create_xgboost_classifier() -> XGBClassifier:
    """Return an XGBClassifier with the hyperparameters used in notebook 04."""
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
