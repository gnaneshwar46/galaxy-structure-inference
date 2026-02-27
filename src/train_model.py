"""
train_model.py

Baseline logistic regression model
with proper scaling via sklearn Pipeline.
"""

import logging
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config_path: str):
    config = load_config(config_path)

    # Load splits
    train = pd.read_csv("data/splits/train.csv")
    test = pd.read_csv("data/splits/test.csv")

    X_train = train.drop(columns=["structural_class"])
    y_train = train["structural_class"]

    X_test = test.drop(columns=["structural_class"])
    y_test = test["structural_class"]

    logging.info(f"Training samples: {X_train.shape[0]}")
    logging.info(f"Testing samples: {X_test.shape[0]}")

    # Build pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Cross-validation
    cv = StratifiedKFold(
        n_splits = config["split"]["n_splits_cv"],
        shuffle = True,
        random_state = config["split"]["random_state"]
    )

    roc_scores = cross_val_score(
        model, 
        X_train,
        y_train,
        cv = cv,
        scoring = "roc_auc"
    )

    bal_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv = cv,
        scoring = "balanced_accuracy"
    )

    print("\n=== Cross-validation (Training Set) ===")
    print(f"Mean ROC-AUC: {roc_scores.mean():.4f} ± {roc_scores.std():.4f} ")
    print(f"Mean Balanced Accuracy: {bal_scores.mean():.4f} ± {bal_scores.std():.4f}")

    # Fit on full training set after CV
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Train
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)
    print("\n=== Baseline Logistic Regression ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

# -----------------------------------------------------------------------------
# Random Forest (Non-linear comparison)
# -----------------------------------------------------------------------------

    rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=config["split"]["random_state"],
    n_jobs=-1
)

    # Cross-validation on training set
    rf_roc_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        cv=cv,
        scoring = "roc_auc"
)

    rf_bal_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        cv=cv,
        scoring = "balanced_accuracy"
)
    
    print("\n=== Random Forest Cross-Validation (Training Set) ===")
    print(f"Mean ROC-AUC: {rf_roc_scores.mean():.4f} ± {rf_roc_scores.std():.4f}")
    print(f"Mean Balanced Accuracy: {rf_bal_scores.mean():.4f} ± {rf_bal_scores.std():.4f}")

    # Fit on full trainig set
    rf_model.fit(X_train, y_train)

    # Evaluate on test set
    rf_y_pred = rf_model.predict(X_test)
    rf_y_prob = rf_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_y_pred)
    rf_bal_acc = balanced_accuracy_score(y_test, rf_y_pred)
    rf_roc = roc_auc_score(y_test, rf_y_prob)

    print("\n=== Random Forest Test Performance ===")
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"Balanced Accuracy: {rf_bal_acc:.4f}")
    print(f"ROC-AUC: {rf_roc:.4f}")

# ------------------------------------------------------------------------
# Feature Importance (Random Forest)
# ------------------------------------------------------------------------

    importances = rf_model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by= "importance", ascending=False)

    print("\n==== Random Forest Feature Importances ===")
    print(importance_df)

if __name__ == "__main__":
    run_training("config.yaml")