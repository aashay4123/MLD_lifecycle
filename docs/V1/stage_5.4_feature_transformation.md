# Transformers for Machine Learning

A deep dive into **feature transformations**—techniques that modify raw feature values to improve model learning.
Categorized into three broad types:

1. **Mathematical Transformations** (simple functions like log, reciprocal, square root, etc.)
2. **Power Transformations** (Box‑Cox, Yeo‑Johnson, quantile)
3. **Scaling & Normalization** (min–max, standard, robust, etc.)

For each, we cover **what** it is, **when to use**, **when not to use**, **common pitfalls**, and **best practices**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Transformations](#mathematical-transformations)

   - 2.1 [Logarithmic Transform](#21-logarithmic-transform)
   - 2.2 [Square-Root & Cube-Root Transforms](#22-square-root--cube-root-transforms)
   - 2.3 [Reciprocal Transform](#23-reciprocal-transform)
   - 2.4 [Exponential Transform](#24-exponential-transform)
   - 2.5 [Power & Polynomial Transforms](#25-power--polynomial-transforms)

3. [Power Transformations](#power-transformations)

   - 3.1 [Box‑Cox Transform](#31-box‑cox-transform)
   - 3.2 [Yeo‑Johnson Transform](#32-yeo‑johnson-transform)
   - 3.3 [Quantile Transform (Rank Mapping)](#33-quantile-transform-rank-mapping)

4. [Scaling & Normalization](#scaling--normalization)

   - 4.1 [Standard (Z‑Score) Scaling](#41-standard-z-score-scaling)
   - 4.2 [Min–Max Scaling](#42-min–max-scaling)
   - 4.3 [Robust Scaling (IQR‑Based)](#43-robust-scaling-iqr-based)
   - 4.4 [MaxAbs Scaling](#44-maxabs-scaling)
   - 4.5 [Normalizer (Vector Normalization)](#45-normalizer-vector-normalization)

5. [Choosing the Right Transformer](#choosing-the-right-transformer)
6. [Key Pitfalls & Best Practices](#key-pitfalls--best-practices)
7. [References](#references)

---

## Introduction

Feature transformations convert raw data into representations that align better with a model’s assumptions or improve its ability to learn. For instance, many algorithms (like linear regression) assume input features are roughly normally distributed; applying a log transform can reduce skewness. Scaling ensures features lie on comparable ranges, preventing dominance by large‑magnitude variables. Power transforms rectify non‑normality, while mathematical transforms can linearize relationships or stabilize variance.

---

## Mathematical Transformations

Simple functions applied directly to a single feature. They often aim to reduce skewness, handle non‑linear relationships, or stabilize variance. Since they are straightforward, they can be implemented without fitting on data (aside from ensuring valid inputs).

### 2.1 Logarithmic Transform

- **Definition:**

  $$
    X' = \log(X + c)
  $$

  where $c\ge0$ (often $c=1$) to handle zeros.

- **When to Use:**

  - **Right‑skewed** distributions (heavy tail to the right).
  - Situations where multiplicative relationships exist (e.g., income, population counts).
  - When variance grows with mean (heteroscedasticity).

- **When _Not_ to Use:**

  - If feature has zeros or negatives without offset (negative values → undefined).
  - Left‑skewed distributions (log will exaggerate low values).
  - Models that are invariant to monotonic transforms (e.g., tree‑based) may not benefit dramatically.

- **Common Pitfalls:**

  - Forgetting to add a small constant: $\log(0)$ is undefined.
  - Over‑transforming small values: if $X$ ∈ \[0,1], $\log$ compresses them heavily.
  - Interpreting coefficients: a one‑unit change in $\log(X)$ corresponds to multiplicative change in $X$.

- **Best Practices:**

  - Check for zeros/negatives. Use $\log1p\,(X) = \log(X + 1)$ for simplicity.
  - Inspect distribution before/after transform (histogram + QQ‑plot).
  - If only slight skew, consider milder transforms (square‑root).
  - Document rationale (e.g., “log‐transform revenue for normality”).

---

### 2.2 Square-Root & Cube-Root Transforms

- **Definition:**

  $$
    X' = \sqrt{X}\quad\text{or}\quad X' = \sqrt[3]{X}.
  $$

- **When to Use:**

  - Moderately **right‑skewed** distributions (less extreme than log).
  - Count data (Poisson‑distributed) where variance \~ mean.
  - To stabilize variance when relationship is sub‑linear.

- **When _Not_ to Use:**

  - Negative or zero values (square root requires $X\ge0$); cube‑root can handle negatives but may not fix skew effectively.
  - If data are highly skewed → log is often more appropriate.

- **Common Pitfalls:**

  - Misinterpreting scale: square‑root reduces scale less aggressively than log but more than no transform.
  - Failing to handle zeros (for $\sqrt$, zero is fine; negative not).
  - If distribution is left‑skewed, transform worsens skew.

- **Best Practices:**

  - Check histogram before/after.
  - Use cube‑root if negative values exist (e.g. returns that can be negative).
  - Compare AIC/BIC or cross‑validated error with/without transform in linear models.

---

### 2.3 Reciprocal Transform

- **Definition:**

  $$
    X' = \frac{1}{X + c}.
  $$

  Often $c=0$ if $X>0$, else $c$ shifts to avoid division by zero.

- **When to Use:**

  - When a feature has a strong **inverse** relationship with the target.
  - If large $X$ values have diminishing returns (e.g., “rooms per square foot”).

- **When _Not_ to Use:**

  - If $X$ can be zero or negative without an offset → division by zero or sign flip.
  - Moderate skew: reciprocal exaggerates differences among small $X$, may over‑distort.

- **Common Pitfalls:**

  - Numerical instability for very small $X$ (e.g., $X=10^{-6}$).
  - Unintended sign flips if $c$ not large enough.
  - Hard to interpret: small changes in big $X$ become extremely small $X'$.

- **Best Practices:**

  - Add $c$ such that $X + c > 0$ for all.
  - Inspect scatterplot against target: if it looks hyperbolic, reciprocal may linearize.
  - Always document the shift $c$.

---

### 2.4 Exponential Transform

- **Definition:**

  $$
    X' = e^X \quad\text{or}\quad X' = \exp(\alpha X)
  $$

  to amplify differences among large values.

- **When to Use:**

  - Rare: when small changes in $X$ correspond to large multiplicative effects.
  - If target responds **exponentially** to feature (e.g. decay processes, compounding growth).

- **When _Not_ to Use:**

  - Most real‑world features are skewed right; exponent will make skew worse.
  - If $X$ already large (e.g. $X>10$), $e^X$ becomes astronomical → overflow.

- **Common Pitfalls:**

  - Numerical overflow for moderate $X$.
  - Exacerbates outliers: a single large $X$ blows up.
  - Difficult to interpret output scale.

- **Best Practices:**

  - Only when domain knowledge suggests an exponential relationship (e.g. half‑life decay).
  - Clip or trim extreme $X$ before exponentiation.
  - Test via scatterplot: if $\log(\text{target})$ vs $X$ is linear, exponent may be appropriate.

---

### 2.5 Power & Polynomial Transforms

- **Definition:**

  - **Power:** $X' = X^\alpha$ for some $\alpha \ne 1$.
  - **Polynomial Features:** create new columns like $X^2, X^3, X_1 \times X_2,$ etc.

- **When to Use:**

  - To capture **non‑linear** relationships explicitly in linear models.
  - If scatterplot of $X$ vs target suggests a quadratic or cubic relationship.

- **When _Not_ to Use:**

  - Overfitting risk: high‑degree polynomials can oscillate wildly.
  - Multicollinearity: powers of $X$ are highly correlated → unstable weights.
  - If using tree models, they can implicitly capture non‑linearity → explicit polynomials not needed.

- **Common Pitfalls:**

  - High‑degree polynomials blow up for large values (numerical instability).
  - Feature explosion: for $d$ original features and degree $p$, combinatorial growth.
  - Ignoring interaction effects: adding only $X^2$ but ignoring cross‑terms when needed.

- **Best Practices:**

  - Limit degree to 2 or 3 only when justified.
  - Use **regularization** (Ridge/Lasso) when adding polynomial terms.
  - Standardize $X$ before raising to powers to reduce numerical issues.
  - Use scikit‑learn’s `PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)`.

---

## Power Transformations

These transforms aim to make the data more **Gaussian** (symmetrical) by choosing an optimal power parameter $\lambda$. They are fitted—i.e., require estimating $\lambda$ from data—unlike fixed mathematical transforms.

### 3.1 Box‑Cox Transform

- **Definition:** (requires $X > 0$)

  $$
    X' =
    \begin{cases}
      \dfrac{X^\lambda - 1}{\lambda}, & \lambda \ne 0, \\
      \log(X), & \lambda = 0.
    \end{cases}
  $$

  $\lambda$ is chosen (e.g. via maximum likelihood) to maximize normality.

- **When to Use:**

  - Positive continuous data that is **right‑skewed**.
  - When a specific $\lambda$ can be estimated robustly (sufficient sample size).
  - Desiring approximate Gaussianity for models sensitive to normality (e.g., linear regression, k‑means).

- **When _Not_ to Use:**

  - If data contain zeros or negative values (Box‑Cox invalid).
  - If distribution is multimodal—transform won’t fix that.
  - If sample size is too small (< 30–50), $\lambda$ estimate unstable.

- **Common Pitfalls:**

  - Fitting $\lambda$ including test data → leakage. Always fit on training only and apply to test.
  - Ignoring zeros: need to shift $X + c$ so minimal value > 0 (shifting changes distribution).
  - Over‑reliance: even if $\lambda \approx 1$, transform does virtually nothing—waste of effort.

- **Best Practices:**

  - Use scikit‑learn’s `PowerTransformer(method="box-cox")` (automatically finds $\lambda$ via MLE).
  - Inspect histograms before/after; check normality (e.g., Shapiro–Wilk).
  - If zeros exist, apply a small shift: $X'' = X + \epsilon$ with $\epsilon = 1e-6$ or minimal positive minus zero.
  - Document chosen $\lambda$ in `preprocessor_manifest.json`.

---

### 3.2 Yeo‑Johnson Transform

- **Definition:** (works for $X \in \mathbb{R}$)

  $$
    X' =
    \begin{cases}
      \dfrac{[(X + 1)^\lambda - 1]}{\lambda}, & X \ge 0,\,\lambda \ne 0, \\
      \log(X + 1), & X \ge 0,\,\lambda = 0, \\
      -\dfrac{[-X + 1]^{2 - \lambda} - 1}{2 - \lambda}, & X < 0,\,\lambda \ne 2, \\
      -\log(-X + 1), & X < 0,\,\lambda = 2.
    \end{cases}
  $$

  $\lambda$ chosen to maximize normality.

- **When to Use:**

  - Data with **both positive and negative** values and skewness.
  - When Box‑Cox not possible due to zeros/negatives.
  - When a more flexible transform is needed to approximate Gaussianity.

- **When _Not_ to Use:**

  - If data distribution is heavily discrete or multimodal (transform won’t cure that).
  - If extreme outliers dominate—transform might not fully normalize.

- **Common Pitfalls:**

  - Over‑shrinkage of negative tail if distribution heavily skewed negative.
  - Instability in $\lambda$ if few negative values exist.
  - Interpreting coefficients: transform no longer simple log or power.

- **Best Practices:**

  - Use scikit‑learn’s `PowerTransformer(method="yeo-johnson")`.
  - Fit on training data only.
  - Visualize distribution post‑transform; run normality tests.
  - If transformation worsens symmetry, reconsider simpler transforms.

---

### 3.3 Quantile Transform (Rank Mapping)

- **Definition:**

  - Map each value to its empirical quantile:

    $$
      X' = \Phi^{-1}(\text{rank}(X)/N)
    $$

    where $\Phi^{-1}$ is the inverse CDF (standard normal), or any chosen distribution (“uniform” outputs uniform \[0,1]).

  - Scikit‑learn’s `QuantileTransformer(output_distribution="normal" or "uniform")`.

- **When to Use:**

  - To enforce a **uniform** or **Gaussian** marginal distribution exactly.
  - When features have arbitrary shapes that simple power transforms can’t fix.
  - When you want to remove outliers’ impact (but preserve ordering).

- **When _Not_ to Use:**

  - If you need to preserve absolute or relative distances—quantile destroys spacing.
  - If dataset is small: quantile estimates unstable.
  - If ranking alone discards important interval information.

- **Common Pitfalls:**

  - **Loss of interpretability**: ranks → no original scale.
  - For new test values outside training range, values are clipped to \[0,1].
  - Discontinuous: ties get assigned identical quantiles → discrete jumps.

- **Best Practices:**

  - Use only on sufficiently large datasets (≥ 500 samples) to get reliable quantiles.
  - If you need a normal output for e.g. Gaussian‑based models (linear, kNN).
  - Fit on train; apply same transformer to test (avoid refitting).
  - Document that raw scale is lost; only ordering preserved.

---

## Scaling & Normalization

These transforms ensure features lie on comparable scales—important for distance‑based or gradient‑based algorithms (e.g., KNN, SVM, neural nets). They are typically **fitted** on training data and **applied** to test.

### 4.1 Standard (Z‑Score) Scaling

- **Definition:**

  $$
    X' = \frac{X - \mu}{\sigma}
  $$

  where $\mu, \sigma$ are training‑set mean and standard deviation.

- **When to Use:**

  - Algorithms that assume **zero‑mean unit‑variance** input:

    - Linear/logistic regression with regularization.
    - PCA, SVM, KNN, neural networks.

- **When _Not_ to Use:**

  - Tree‑based models (e.g., Random Forest, Decision Trees, XGBoost) are invariant to monotonic scaling.
  - If feature has heavy outliers → mean and std dev are distorted → consider robust scaling.

- **Common Pitfalls:**

  - **Data leakage**: computing $\mu,\sigma$ on full dataset including test.
  - Outliers skew $\mu,\sigma$, leading to compressed bulk of data.

- **Best Practices:**

  - Fit `StandardScaler` on training set only; use same `transform` on test.
  - Inspect distribution; if heavy outliers, consider `RobustScaler` instead.
  - Store fitted $\mu,\sigma$ in `preprocessor_manifest.json` (for reproducibility).

---

### 4.2 Min–Max Scaling

- **Definition:**

  $$
    X' = \frac{X - \min(X)}{\max(X) - \min(X)}
  $$

  → maps $X$ into $[0,1]$. Variants exist for mapping to $[-1,1]$.

- **When to Use:**

  - When features need to be bounded (e.g., image pixel values).
  - For algorithms sensitive to feature range but not assumption of normality (e.g., neural nets, KNN).
  - If you want interpretable bounds.

- **When _Not_ to Use:**

  - If outliers exist: $\min,\max$ become outlier‑driven, compressing majority into small range.
  - For algorithms requiring zero‑mean (e.g., PCA) → prefer standard scaling.

- **Common Pitfalls:**

  - New test data outside original $[\min,\max]$ clipped to \[0,1] → may distort.
  - No mechanism for robust handling of outliers.

- **Best Practices:**

  - Fit on train; store $\min,\max$.
  - Clip test values to train’s $[\min,\max]$ or allow beyond-range if acceptable.
  - Combine with outlier removal/robust transforms beforehand if needed.

---

### 4.3 Robust Scaling (IQR‑Based)

- **Definition:**

  $$
    X' = \frac{X - \text{median}(X)}{\text{IQR}(X)},
    \quad \text{where IQR} = Q_3 - Q_1.
  $$

  (scikit‑learn’s `RobustScaler` uses median and IQR by default).

- **When to Use:**

  - Presence of **outliers**; median/IQR are robust to extreme values.
  - When you want a scale similar to standard scaling but resistant to tail effects.

- **When _Not_ to Use:**

  - If distribution is not roughly symmetric around median (IQR may not center data well).
  - If algorithm expects zero‑mean (RobustScaling yields zero‐median but not zero‐mean).

- **Common Pitfalls:**

  - Does not guarantee zero‑mean; some algorithms prefer zero‑mean.
  - If 50% of data are identical (IQR=0), scaling fails; need fallback.

- **Best Practices:**

  - Fit on training only; apply to test.
  - If IQR=0 for a feature (constant or quasi‑constant), drop or apply fallback (e.g., divide by 1).
  - Inspect post‑scaling distribution: should see fewer outliers.

---

### 4.4 MaxAbs Scaling

- **Definition:**

  $$
    X' = \frac{X}{\max(|X|)}.
  $$

  Scales features to $[-1,1]$ by dividing by maximum absolute value.

- **When to Use:**

  - Sparse data (e.g., TF‑IDF, one‑hot): preserves sparsity (doesn’t center data).
  - When negative values exist and you want to keep zero‑point.

- **When _Not_ to Use:**

  - If distribution extremely heavy‑tailed: max may be dominated by single outlier.
  - If algorithm expects zero‑mean (e.g., PCA).

- **Common Pitfalls:**

  - Outlier becomes 1 or ‑1; all other values compressed.
  - No guarantee of zero‑centering.

- **Best Practices:**

  - Use on sparse matrices where centering is infeasible.
  - If a feature has a single large outlier, consider clipping or robust scaling first.

---

### 4.5 Normalizer (Vector Normalization)

- **Definition:**
  Scales each **row** (sample) rather than each column:

  $$
    \mathbf{x}' = \frac{\mathbf{x}}{\|\mathbf{x}\|_p},
    \quad p = 1\text{ or }2 \;\text{(L1 or L2 norm)}.
  $$

- **When to Use:**

  - Text/data represented as frequency vectors (e.g., TF, TF‑IDF).
  - Distance‑based methods (KNN, K‑Means) when magnitude variation across samples is undesirable.

- **When _Not_ to Use:**

  - Features representing absolute quantities (e.g., age) where relative proportions meaningless.
  - If row sums are already normalized or features are not comparable.

- **Common Pitfalls:**

  - Weak features become inflated if other features in row are zero.
  - Ignores scale differences across columns.

- **Best Practices:**

  - Apply when sample vectors represent proportions or counts.
  - Choose L2 for Euclidean-based algorithms, L1 for Manhattan-based.
  - Fit doesn’t apply (stateless); simply transform.

---

## Choosing the Right Transformer

| Scenario                                                       | Recommended Transformer(s)                                                                                                                    |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mild right skew**                                            | Logarithmic transform (with shift) or square‑root.                                                                                            |
| **Severe right skew, strictly positive**                       | Box‑Cox.                                                                                                                                      |
| **Skew + negative values**                                     | Yeo‑Johnson.                                                                                                                                  |
| **Ensuring approximate normality for linear models**           | Box‑Cox (if positive) or Yeo‑Johnson (if negative allowed) → then Standard Scaling.                                                           |
| **Count data (zero‐inflated)**                                 | Add 1 → log1p, or square‑root.                                                                                                                |
| **Data with heavy outliers (wide tail)**                       | RobustScaler for scaling; consider log or power transform to compress tails.                                                                  |
| **Features on vastly different scales (e.g. income vs age)**   | StandardScaler or Min–Max if no outliers; RobustScaler if outliers exist.                                                                     |
| **Sparse high‑dimensional (e.g. text, one‑hot)**               | MaxAbsScaler (preserves sparsity), or no scaling for tree models; Normalizer if comparing samples by direction only (e.g. cosine similarity). |
| **Non‑linear relationships suspected**                         | PolynomialFeatures (degree 2 or 3) in a regularized pipeline; OR compute interaction terms manually.                                          |
| **Rank‑based normality enforcement**                           | QuantileTransformer (output_distribution=“normal”), especially for algorithms assuming Gaussian marginal (kNN, k‑means).                      |
| **Categorical numeric features to be treated as “continuous”** | Often no direct mathematical transform; consider binning, target encoding (separate section).                                                 |
| **Streaming/online transforms**                                | Yeo‑Johnson or Box‑Cox fitted on initial batch—subsequent data use fixed $\lambda$. Normalizer for each incoming sample.                      |

---

## Key Pitfalls & Best Practices

- **Data Leakage:**
  Always **fit** transformers (e.g., computing $\lambda$ for Box‑Cox or $\mu,\sigma$ for StandardScaler) on **training data only**. Use the same parameters to transform validation/test sets.

- **Handling Zero/Negative Values:**

  - Box‑Cox requires strictly positive inputs → shift or use Yeo‑Johnson instead.
  - Log transform: apply `log1p` or add a small constant to avoid $\log(0)$.
  - Reciprocal: add constant to avoid division by zero.

- **Outliers:**

  - StandardScaler is sensitive to outliers → consider RobustScaler or clipping before scaling.
  - Extremely large outliers can make power transforms produce extreme values → handle or remove outliers first.

- **Interpretability:**

  - Mathematical transforms (log, sqrt) are easier to justify and interpret in physics/economics contexts.
  - Power transforms (Box‑Cox, Yeo‑Johnson) lose interpretability of raw scale → note in documentation.

- **Numerical Stability:**

  - Polynomial features: scale features first to reduce numerical error when raising to high powers.
  - QuantileTransformer: for small datasets, quantile estimation can be noisy → only use on sufficiently large datasets.

- **Feature Engineering Workflow:**

  1. **EDA →** examine feature distributions (histograms, skew, kurtosis).
  2. **Select Transform →** choose appropriate transform to reduce skew or stabilize variance.
  3. **Fit on Train →** derive parameters ($\lambda$, $\mu,\sigma$, $\min,\max$).
  4. **Transform Train/Val/Test →** apply same parameters.
  5. **Modeling →** inspect residuals / feature importances; possibly revisit transformations.

- **Document Everything:**

  - Record chosen transformation and parameters in a manifest (e.g., `prep_manifest.json`).
  - Note why a transform was applied (e.g., “Log-transform revenue because right-skewed and improved normality from skew=2.5 → 0.3”).
  - Ensure reproducibility: include exact code (e.g., `PowerTransformer(method="box-cox", standardize=False)`).

---

## References

1. Altman, N. S., & Krzywinski, M. (2017). Points of significance: Transforming data. _Nature Methods_, 14, 119–120.
2. Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. _Journal of the Royal Statistical Society. Series B (Methodological)_, 26(2), 211–252.
3. Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. _Biometrika_, 87(4), 954–959.
4. Ghasemi, A., & Zahediasl, S. (2012). Normality tests for statistical analysis: a guide for non-statisticians. _International Journal of Endocrinology and Metabolism_, 10(2), 486–489.
5. Pedregosa, F., et al. (2011). Scikit‑learn: Machine Learning in Python. _Journal of Machine Learning Research_, 12, 2825–2830.
