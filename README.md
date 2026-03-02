# Galaxy Structure Inference  
## Testing Whether Structure Emerges from Multivariate Interaction

---

## Scientific Motivation

Galaxy morphology (disk-dominated vs bulge-dominated) is strongly correlated with global galaxy properties such as stellar mass and size.

This project investigates the hypothesis:

> Galaxy structure emerges from multivariate interaction between mass, size, and cosmic context rather than from a single dominant parameter.

We test this using controlled machine learning experiments on the NASA-Sloan Atlas (NSA) low-redshift galaxy sample.

---

## Dataset

**Source:** NASA-Sloan Atlas (NSA)  
**Redshift range:** $z < 0.08$ (conservative structural reliability cut)

Extracted physical quantities:

- Stellar mass ($M_*$; Sérsic-based)
- Effective radius ($R_e$; Sérsic half-light radius)
- Spectroscopic redshift ($z$)

The Sérsic index ($n$) is used only to define structural class and is removed from the feature set to prevent target leakage.

---

## Experimental Design

### 1. Controlled Preprocessing

- FITS ingestion with endian correction  
- Conservative redshift filtering  
- Removal of non-physical values  
- Log-transform of physical scale quantities  
- Reproducible configuration-driven pipeline  

---

### 2. Structural Classification

Binary label defined as:

- Disk-dominated: $n < 2.5$  
- Bulge-dominated: $n \ge 2.5$  

Stratified 80/20 train-test split ensures class balance preservation.

---

## Baseline Model: Logistic Regression (Linear)

**Cross-validated performance (5-fold):**

- ROC-AUC $\approx 0.842 \pm 0.001$  
- Balanced Accuracy $\approx 0.789 \pm 0.001$  

**Test set:**

- ROC-AUC $\approx 0.840$  
- Balanced Accuracy $\approx 0.787$  

This indicates that mass–size scaling captures most structural separation linearly.

---

## Non-Linear Comparison: Random Forest

**Cross-validated performance:**

- ROC-AUC $\approx 0.880 \pm 0.001$  
- Balanced Accuracy $\approx 0.803 \pm 0.001$  

**Test set:**

- ROC-AUC $\approx 0.878$  
- Balanced Accuracy $\approx 0.803$  

Non-linear modeling provides measurable improvement over the linear baseline, indicating additional structural interactions beyond simple linear scaling.

---

## Feature Importance (Random Forest)

Relative contribution:

- Stellar Mass ($\sim 48\%$)  
- Effective Radius ($\sim 32\%$)  
- Redshift ($\sim 20\%$)  

This suggests structure is primarily encoded in mass–size scaling, with non-linear interaction contributing additional predictive power.

---

## Compactness Hypothesis Test

Surface stellar mass density was engineered:

$$
\Sigma_* = \frac{M_*}{R_e^2}
$$

Taking logarithms:

$$
\log \Sigma_* = \log M_* - 2 \log R_e
$$

Adding this feature to the linear model did not improve performance, confirming that compactness alone does not explain the non-linear gain in a purely linear framework.

However, permutation importance analysis revealed that surface density causes the largest ROC-AUC drop when shuffled.

This indicates that compactness is the dominant physical driver, but its role emerges through interaction rather than as a single isolated feature.

---

# Mass–Size Geometric Boundary Analysis

To directly interpret the learned structural decision boundary, logistic regression coefficients were extracted and the implied mass–size tradeoff slope was computed.

The decision boundary in log-space satisfies:

$$
R_e = -\frac{\beta_M}{\beta_R} M_* + C
$$

Empirically:

$$
-\frac{\beta_M}{\beta_R} \approx 2.01
$$

This implies that the morphology transition approximately satisfies:

$$
\log M_* - 2 \log R_e = \text{constant}
$$

which corresponds to a stellar surface mass density threshold.

More generally, the boundary can be written as:

$$
\log M_* - \alpha \log R_e = \text{constant}
$$

with

$$
\alpha \approx 2
$$

---

## Robustness Tests

The geometric boundary was tested extensively:

- Cross-validation slope: $2.008 \pm 0.003$  
- Regularization sensitivity ($C = 0.1$–$10$): slope stable to four decimal places  

This demonstrates that the mass–size boundary geometry is intrinsic to the data and not a numerical artifact.

---

# Redshift Evolution of Structural Threshold

To test for cosmic evolution, the sample was divided into three equal-sized redshift bins.

Mass–size boundary slopes:

- Low $z$: $\sim 2.40$  
- Mid $z$: $\sim 1.79$  
- High $z$: $\sim 1.59$  

A linear fit yields:

$$
\frac{d(\text{slope})}{dz} \approx -18.8
$$

This indicates that the morphology transition becomes increasingly compactness-regulated toward lower redshift.

The structural boundary is therefore not strictly static across cosmic time.

---

# Refined Interpretation

1. Galaxy morphology is strongly encoded in mass–size geometry.  
2. The structural transition closely resembles a surface-density threshold.  
3. This geometric boundary is extremely stable under cross-validation and regularization changes.  
4. The boundary rotates systematically with redshift, suggesting mild evolution in structural quenching behavior.  
5. Compactness emerges as a derived geometric consequence rather than a manually engineered dominant feature.

Overall, the results support the hypothesis that galaxy structure emerges from multivariate physical interaction rather than from a single linear parameter threshold.

---

# Reproducibility

This project is fully reproducible from raw data to model evaluation.

### 1. Install Dependencies

Create and activate a virtual environment, then install:

- pandas  
- numpy  
- scikit-learn  
- astropy  
- pyyaml  

### 2. Prepare Raw Data

Place the NSA FITS file inside the appropriate data directory.

Raw data is not version-controlled to keep the repository lightweight.

### 3. Run Data Cleaning

This step:

- Loads FITS data  
- Applies conservative redshift filtering ($z < 0.08$)  
- Removes non-physical entries  
- Applies log-transformation to scale quantities  
- Saves cleaned dataset  

### 4. Create Stratified Train/Test Split

This step:

- Creates binary structural label ($n$ threshold = 2.5)  
- Removes Sérsic index from features (leakage prevention)  
- Performs stratified 80/20 split  

### 5. Train and Evaluate Models

This script:

- Performs 5-fold stratified cross-validation  
- Trains logistic regression baseline  
- Trains Random Forest for non-linear comparison  
- Extracts geometric boundary slope  
- Tests slope stability  
- Evaluates redshift evolution  

---

# Project Philosophy

This repository emphasizes:

- Hypothesis-driven experimentation  
- Strict leakage prevention  
- Controlled statistical validation  
- Geometric interpretability of learned boundaries  
- Robustness testing  
- Physically meaningful interpretation  

The objective is not model maximization, but scientific understanding of how galaxy structure emerges from global physical properties.

---

# Current Status

- Low-redshift structural sample established (~287k galaxies)  
- Linear baseline validated  
- Non-linear comparison performed  
- Compactness hypothesis tested  
- Mass–size geometric boundary extracted  
- Slope stability validated  
- Redshift evolution detected  

This project is complete as a portfolio-level structural inference study.
