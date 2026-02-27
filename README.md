# Galaxy Structure Inference
### Testing Whether Structure Emerges from Multivariate Interaction

----

## Scientific Motivation

Galaxy morphology (disk-dominated vs bulge-dominated) is strongly correlated with global galaxy properties such as stellar mass and size.

This project investigates the hypothesis:

> **Galaxy structure emerges from multivariate interaction between mass, size, and cosmic context rather than from a single dominant parameter.**

We test this using controlled machine learning experiments on the NASA-Sloan Atlas (NSA) low-redshift galaxy sample.

----

## Dataset

- Source: NASA-Sloan Atlas (NSA)
- Redshift range: **z < 0.08** (conservative structural reliability cut)

Extracted physical quantities:
- Stellar mass (Sérsic-based)
- Effective radius (Sérsic half-light radius)
- Spectroscopic redshift

The Sérsic index is used **only to define structural class** and is removed from features to prevent target leakage.

----

## Experimental Design

### 1. Controlled Preprocessing
- FITS ingestion with endian correction
- Conservative redshift filtering
- Removal of non-physical values
- Log-transform of physical scale quantities
- Reproducible configuration-driven pipeline

### 2. Structural Classification
Binary label defined as:

- Disk-dominated: n < 2.5
- Bulge-dominated: n ≥ 2.5

Stratified 80/20 train-test split ensures class balance preservation.

----

## Baseline Model: Logistic Regression (Linear)

Cross-validated performance (5-fold):

- ROC-AUC ≈ 0.842 ± 0.001
- Balanced Accuracy ≈ 0.789 ± 0.001

Test set:

- ROC-AUC ≈ 0.840
- Balanced Accuracy ≈ 0.787

This indicates that mass-size scaling captures most structural separation linearly.

----

## Non-Linear Comparison: Random Forest

Cross-validated performance:

- ROC-AUC ≈ 0.880 ± 0.001
- Balanced Accuracy ≈ 0.803 ± 0.001

Test set:

- ROC-AUC ≈ 0.878
- Balanced Accuracy ≈ 0.803

Non-linear modeling provides measureable improvement over the linear baseline.

----

## Feature Importance (Random Forest)

Relative contribution:

1. Stellar Mass (~48%)
2. Effective Radius (~32%)
3. Redshift (~20%)

This suggests structure is primarily encoded in mass-size scaling, with non-linear interaction contributing additional predictive power.

----

## Current Conclusion

### 1. Linear Baseline

Logistic regression achieves ROC-AUC ≈ 0.84 using mass, size, and redshift.

This indicates that galaxy structure is largely separable in mass-size space using a linear decision boundary.

----

### 2. Non-Linear Gain

Random Forest improves performance to ROC-AUC ≈ 0.875.

This demonstrates that additional non-linear interactions contribute measureable predictive power beyond simple linear scaling.

----

### 3. Compactness Hypothesis Test

Surface stellar mass density was engineered:

log Σ_* = log M_* − 2 log R_e

Adding this feature to the linear model did **not** improve performance, comfirming that compactness alone does not explain the non-linear gain in a purely linear framework.

However, permutation importance analysis revealed that surface density causes the largest ROC-AUC drop when shuffled.

This indicates that compactness is the dominant physical driver, but in a non-linear manner.

----

### Refined Interpretation

- Galaxy structure is strongly encoded in compactness.
- The dependence is not purely linear.
- Non-linear coupling between mass and size provides additional explanatory power.
- Redshift contributes secondary contextual information in the low-redshift regime.

Overall, the results support the hypothesis that galaxy structure emerges from multivariate physical interaction rather than from a single linear parameter threshold.

## Reproducibility

This project is fully reproducible from raw data to model evaluation.

### 1. Install Dependencies

Create and activate a virtual environment, then install:
- pandas
- numpy
- scikit-learn
- astropy 
- pyyaml

----

### 2. Prepare Raw Data

Place the NSA FITS file inside:

Raw data is not version-controlled to keep the repository lightweight.

----

### 3. Run Data cleaning

This step:

- Loads FITS data
- Applies conservative redshift filtering (z < 0.08)
- Removes non-physical entries
- Applies log-transformation to scale quantities
- Saves cleaned dataset

----

### 4. Create Stratified Train/Test Split

This step:
- Creates binary structural label (n threshold = 2.5)
- Removes Sérsic index from features (leakage prevention)
- Performs stratified 80/20 split.

----

### 5. Train and Evaluate Models 

This scripts:
- Performs 5-fold stratified cross-validation
- Trains logistic regression baseline
- Trains Random Forest for non-linear comparison
- Prints feature importances

---- 

## Project Philosophy

This repository emphasizes:

- Hypothesis-driven experimentation
- Strict leakage prevention
- Controlled statistical validation
- Clear separation of preprocessing and modeling
- Physically interpretable results

The objective is not model maximization, but scientific understanding of how galaxy structure emerges from global physical properties.

----

## Current Status

- Low-redshift structural sample established (~287k galaxies)
- Linear baseline validated and cross-validated
- Non-linear comparison performed
- Interaction effects detected
- Feature importance interpreted physically

Next steps focus on deeper physical feature interpretation rather than performance tuning.

----