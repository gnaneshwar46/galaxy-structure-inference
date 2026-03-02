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
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

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

    # ---------------------------------------------------------
    # Mass-Size geometric model
    # Keep stellar_mass, effective_radius, redshift
    # Remove surface_density
    # ---------------------------------------------------------

    X_train = X_train.drop(columns=["surface_density"])
    X_test = X_test.drop(columns=["surface_density"])

    logging.info("Running mass-size geometric model (M*, Re + z).")

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

# ----------------------------------------------------------------------------
# Logistic Coefficients (Structural Interpretation)
# ----------------------------------------------------------------------------

    coef = model.named_steps["classifier"].coef_[0]
    features = X_train.columns

    print("\n=== Logistic Coefficients ===")
    for f, c in zip(features, coef):
        print(f"{f}: {c:.6f}")

    # Compute mass-size tradeoff slope

    beta_mass = coef[list(features).index("stellar_mass")]
    beta_radius = coef[list(features).index("effective_radius")]

    slope = -beta_mass / beta_radius
    print(f"\nMass-Size Boundary Slope (d logR / d logM): {slope:.4f}")

# -----------------------------------------------------------------------------
# 2D Mass-Size Probability Surface (Logistic Regression)
# -----------------------------------------------------------------------------

    # Get trained logistic model
    log_model = model

    # Define grid range from training data
    mass_min, mass_max = X_train["stellar_mass"].min(), X_train["stellar_mass"].max()
    re_min, re_max = X_train["effective_radius"].min(), X_train["effective_radius"].max()

    mass_grid = np.linspace(mass_min, mass_max, 100) 
    re_grid = np.linspace(re_min, re_max, 100)

    M, R = np.meshgrid(mass_grid, re_grid)

    # Fix redshift at median
    z_fixed = X_train["redshift"].median()

    # Ensure correct column order from training
    feature_order = X_train.columns

    grid_df = pd.DataFrame({
        "stellar_mass": M.ravel(),
        "effective_radius": R.ravel(),
        "redshift": z_fixed
    })

    # Reorder explicitly
    grid_df = grid_df[feature_order]

    # Predict probability
    probs = log_model.predict_proba(grid_df)[:, 1]
    print("Mass range:", X_train["stellar_mass"].min(), X_train["stellar_mass"].max())
    print("Radius range:", X_train["effective_radius"].min(), X_train["effective_radius"].max())
    print("Probability range:", probs.min(), probs.max())

    Z = probs.reshape(M.shape)

    # Plot
    plt.figure(figsize = (8, 6))

    contour = plt.contourf(
        M,
        R,
        Z,
        levels = np.linspace(0, 1, 20),
        vmin = 0,
        vmax = 1
    )

    plt.xlabel("log Stellar Mass")
    plt.ylabel("log Effective Radius")
    plt.title("Morphology Probability Surface (Logistic)")
    plt.colorbar(contour, label = "P(Early-type)")
    plt.tight_layout()
    plt.savefig("figures/mass_size_probability_surface.png", dpi = 300)
    plt.close()

    logging.info("Saved 2D mass-size probability surface.")

# ----------------------------------------------------------------------------
# Mass-Size Slope Stability Across CV Folds
# ----------------------------------------------------------------------------

    print("\n=== Slope Stability Across CV Folds ===")

    slopes = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter = 1000))
        ])

        fold_model.fit(X_tr, y_tr)

        coef_fold = fold_model.named_steps["classifier"].coef_[0]
        features_fold = X_tr.columns

        beta_mass = coef_fold[list(features_fold).index("stellar_mass")]
        beta_radius = coef_fold[list(features_fold).index("effective_radius")]

        slope_fold = -beta_mass / beta_radius
        slopes.append(slope_fold)

        print(f"Fold slope: {slope_fold:.4f}")

    slopes = np.array(slopes)

    print(f"\nMean slope: {slopes.mean():.4f}")
    print(f"Std slope: {slopes.std():.4f}")

# -----------------------------------------------------------------------------
# Redshift-Binned Mass-Size Slope Test
# -----------------------------------------------------------------------------

    print("\n=== Redshift-Binned Mass-Size Slopes ===")

    # Create 3 redshift bins (low, mid, high)
    z_bins = np.quantile(X_train["redshift"], [0.0, 0.33, 0.66, 1.0])

    for i in range(3):
        z_min, z_max = z_bins[i], z_bins[i+1]

        mask = (X_train["redshift"] >= z_min) & (X_train["redshift"] <= z_max)
        X_bin = X_train[mask]
        y_bin = y_train[mask]

        print(f"Redshift bin {i + 1} size: {len(X_bin)}")

        bin_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter = 1000))
        ])

        bin_model.fit(X_bin, y_bin)

        coef_bin = bin_model.named_steps["classifier"].coef_[0]
        features_bin = X_bin.columns

        beta_mass = coef_bin[list(features_bin).index("stellar_mass")]
        beta_radius = coef_bin[list(features_bin).index("effective_radius")]

        slope_bin = -beta_mass / beta_radius

        print(f"Redshift bin {i+1} ({z_min:.4f} - {z_max:.4f}) slope: {slope_bin:.4f}")
        
# -----------------------------------------------------------------------------
# Linear Trend of Slope vs Redshift
# -----------------------------------------------------------------------------

    print("\n=== Linear Trend: Slope vs Redshift ===")

    bin_medians = []
    bin_slopes = []

    for i in range(3):
        z_min, z_max = z_bins[i], z_bins[i + 1]
        mask = (X_train["redshift"] >= z_min) & (X_train["redshift"] <= z_max)

        z_median = X_train.loc[mask, "redshift"].median()
        bin_medians.append(z_median)

        # recompute slope (reuse stored slope if you prefer)

        X_bin = X_train[mask]
        y_bin = y_train[mask]

        bin_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter = 1000))
        ])
        bin_model.fit(X_bin, y_bin)

        coef_bin = bin_model.named_steps["classifier"].coef_[0]
        features_bin = X_bin.columns
        
        beta_mass = coef_bin[list(features_bin).index("stellar_mass")]
        beta_radius = coef_bin[list(features_bin).index("effective_radius")]

        slope_bin = -beta_mass / beta_radius
        bin_slopes.append(slope_bin)

    # Fit linear model
    z_array = np.array(bin_medians)
    slope_array = np.array(bin_slopes)

    trend_coef = np.polyfit(z_array, slope_array, 1)

    print(f"Slope vs z linear coefficient: {trend_coef[0]:.4f}")
    print(f"Intercept: {trend_coef[1]:.4f}")

# -----------------------------------------------------------------------------
# Plot: Slope vs Redshift
# -----------------------------------------------------------------------------

    plt.figure(figsize = (6, 5))
    
    plt.scatter(z_array, slope_array)

    # Plot linear fit
    z_fit = np.linspace(z_array.min(), z_array.max(), 100)
    slope_fit = trend_coef[0] * z_fit + trend_coef[1]
    plt.plot(z_fit, slope_fit)

    plt.xlabel("Redshift")
    plt.ylabel("Mass-Size Boundary Slope")
    plt.title("Evolution of Mass-Size Slope with Redshift")
    plt.tight_layout()
    plt.savefig("figures/slope_vs_redshift.png", dpi = 300)
    plt.close()

    logging.info("Saved slope vs redshift figure.")

# ----------------------------------------------------------------------------
# Regularization Sensitivity Test
# ----------------------------------------------------------------------------

    print("\n=== Regularization Sensitivity Test ===")

    C_values = [0.1, 1.0, 10.0]

    for C_val in C_values:
        reg_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter = 1000, C = C_val))
        ])

        reg_model.fit(X_train, y_train)

        coef_reg = reg_model.named_steps["classifier"].coef_[0]
        features_reg = X_train.columns

        beta_mass = coef_reg[list(features_reg).index("stellar_mass")]
        beta_radius = coef_reg[list(features_reg).index("effective_radius")]

        slope_reg = -beta_mass / beta_radius

        print(f"C = {C_val:.1f} -> slope = {slope_reg:.4f}")

# -----------------------------------------------------------------------------
# Random Forest (Non-linear comparison)
# -----------------------------------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=100,
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

    # ----------------------------------------------------
    # Permutaion Importance (Test Set)
    # ----------------------------------------------------

    def roc_auc_from_model(estimator, X, y):
        y_prob = estimator.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_prob)

    perm_importance = permutation_importance(
        rf_model,
        X_test,
        y_test,
        scoring = roc_auc_from_model,
        n_repeats= 5,
        random_state = config["split"]["random_state"],
        n_jobs= -1
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    
    print("\n=== Permutation Importance (Test ROC-AUC Drop) ===") 
    print(perm_df)

if __name__ == "__main__":
    run_training("config.yaml")