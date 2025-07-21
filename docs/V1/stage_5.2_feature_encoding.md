# Encoders for Machine Learning

A comprehensive guide to categorical encoding techniques in classical ML.
Categorized into **Unsupervised Encoders** (no target information used) and **Supervised Encoders** (leverage the target during encoding). For each, we cover when to use, when not to use, common pitfalls, and best practices.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Unsupervised Encoders](#unsupervised-encoders)

   - 2.1 [Label / Ordinal Encoding](#21-label--ordinal-encoding)
   - 2.2 [One-Hot Encoding](#22-one-hot-encoding)
   - 2.3 [Frequency / Count Encoding](#23-frequency--count-encoding)
   - 2.4 [Hashing Encoding](#24-hashing-encoding)

3. [Supervised Encoders](#supervised-encoders)

   - 3.1 [Target (Mean) Encoding](#31-target-mean-encoding)
   - 3.2 [Leave-One-Out Encoding (LOO)](#32-leave-one-out-encoding-loo)
   - 3.3 [K‑Fold (Out-of-Fold) Target Encoding](#33-k‐fold-out-of-fold-target-encoding)
   - 3.4 [Bayesian-Smoothed Encoding](#34-bayesian-smoothed-encoding)
   - 3.5 [James–Stein (Shrinkage) Encoding](#35-james–stein-shrinkage-encoding)
   - 3.6 [CatBoost‑Style Ordered Target Encoding](#36-catboost‑style-ordered-target-encoding)

4. [Choosing the Right Encoder](#choosing-the-right-encoder)
5. [Key Pitfalls & Best Practices](#key-pitfalls--best-practices)
6. [References](#references)

---

## Introduction <a name="introduction"></a>

Categorical features must be converted to numerical form before feeding them into most classical machine-learning algorithms. The choice of encoding method can significantly affect model performance, especially when categories have high cardinality (many unique labels) or when target leakage is a concern. This document divides encoding approaches into:

- **Unsupervised Encoders:** Use only feature values (no target information).
- **Supervised Encoders:** Leverage the target variable to create encodings (risking leakage if done improperly).

---

## Unsupervised Encoders <a name="unsupervised-encoders"></a>

These methods do **not** use the target label when encoding. They are risk‑free from leakage but may not capture any predictive power that resides in the target relationship.

### 2.1 Label / Ordinal Encoding <a name="21-label--ordinal-encoding"></a>

- **Description:**
  Assigns each unique category an integer label.
  Example: `{"Maharashtra": 0, "Delhi": 1, "Karnataka": 2, …}`.

- **When to Use:**

  - Tree‑based models (e.g. Random Forest, XGBoost) that can handle integer splits on categorical levels.
  - Low‑cardinality features (< 50 unique levels) where dimensionality is manageable.

- **When _Not_ to Use:**

  - Linear/logistic models: they interpret integers as ordinal (e.g. “2 > 1”), introducing spurious ordering where none exists.
  - High‑cardinality features (hundreds or thousands of levels) → single integer per category doesn’t capture similarity structure; plus risk of overfitting.

- **Common Pitfalls:**

  - Swapping to a linear model later causes the algorithm to treat “5” as larger than “2.”
  - If new categories appear at inference, you need a fallback (e.g. map unseen → a special integer).

- **Best Practices:**

  - Reserve for tree models only.
  - Map unseen test categories to a reserved index (e.g. `-1`).
  - If cardinality is extremely large, consider hashing or supervised encoders instead.

---

### 2.2 One-Hot Encoding <a name="22-one-hot-encoding"></a>

- **Description:**
  Create a binary column for each unique category.
  Example: `State_Maharashtra`, `State_Delhi`, `State_Karnataka`, … each taking 0/1.

- **When to Use:**

  - Low-to-moderate cardinality (≤ 50–100 distinct levels).
  - Models that benefit from sparse, orthogonal representations (e.g. linear, logistic, tree).

- **When _Not_ to Use:**

  - Very high-cardinality features, which lead to huge sparse matrices (memory blowup).
  - If many categories occur very rarely → many near-zero columns.

- **Common Pitfalls:**

  - Dimensionality explosion: hundreds/thousands of dummy variables.
  - If one-hot columns are collinear (no drop), can cause multicollinearity in linear models → drop one category or use `drop='first'`.
  - Rare categories still get a placeholder column with near‑zero variance.

- **Best Practices:**

  - Drop one column (or use `drop='first'`) to avoid perfect multicollinearity.
  - Limit one-hot to features with ≤ 50 levels; group rare categories into “**other**.”
  - Use sparse representations (`sparse=True` in many libraries).

---

### 2.3 Frequency / Count Encoding <a name="23-frequency--count-encoding"></a>

- **Description:**
  Replace each category with its frequency (proportion) in the training set, or raw counts.
  Example: If “Delhi” appears 10% of the time, encode all “Delhi” rows as 0.10 (or as count=100).

- **When to Use:**

  - Moderate-to-high cardinality when you want a single numeric column.
  - Quick baseline: adds no risk of target leakage.

- **When _Not_ to Use:**

  - If category frequency has no predictive power (e.g. uniform distribution).
  - If you later use a model that can’t differentiate between categories with identical frequency.

- **Common Pitfalls:**

  - Two distinct categories with equal frequency collapse to the same value → loss of discriminative power.
  - Frequency can shift over time; if retraining data differs, encodings drift.

- **Best Practices:**

  - Combine with group-based features or use as a supplementary column.
  - When retraining, recalculate frequencies on new training set to avoid mismatch.
  - Consider smoothing (e.g. combine with global mean) if counts are small.

---

### 2.4 Hashing Encoding <a name="24-hashing-encoding"></a>

- **Description:**
  Map each category (string) to an index in a fixed hash space of size `n_buckets`. Then, either:

  - Keep only the bucket index (as integer).
  - Compute bucket-wise frequency or target statistic.

- **When to Use:**

  - Extremely high cardinality (> 10k levels) where storing a dictionary is expensive.
  - Streaming contexts where categories keep evolving.

- **When _Not_ to Use:**

  - When perfect fidelity of categories is required (hash collisions mix categories).
  - If interpretability of individual categories is important.

- **Common Pitfalls:**

  - **Hash collisions**: Two distinct categories map to the same bucket → ambiguity.
  - Need to choose a bucket size that balances memory vs. collision risk.

- **Best Practices:**

  - Use a bucket size that is several times larger than unique categories in training (e.g. 2×–5×).
  - If using raw bucket index as a feature, combine with a supervised statistic (e.g. bucket target mean).
  - Always use a consistent hash seed (default libraries often fix this).

---

## Supervised Encoders <a name="supervised-encoders"></a>

These encoders **leverage the target label** to construct more predictive features. They can provide strong signal, especially for high‑cardinality categories, but must be used carefully to avoid **data leakage**. When done properly (with cross-validation or leave‑one‑out), they can dramatically improve performance on linear‑type models or small datasets.

### 3.1 Target (Mean) Encoding <a name="#31-target-mean-encoding"></a>

- **Description:**
  Replace each category with the **average target** value for that category, computed on the training set.

  - For classification (binary): encode as proportion of positives (e.g. if `Delhi` churn rate = 0.3, encode “Delhi→0.3”).
  - For regression: encode as mean of target (e.g. average revenue).

- **When to Use:**

  - High‑cardinality categorical features where one-hot is infeasible.
  - Linear or tree models that can exploit a continuous relationship.

- **When _Not_ to Use:**

  - Without any safeguards (CV/LOO), this leaks target info → overly optimistic performance.
  - If categories have very few samples: the mean is unreliable (overfitting risk).

- **Common Pitfalls:**

  - **Target leakage**: Using the same category mean computed on the entire dataset (train+val+test) → model trains on leak.
  - Categories with tiny counts → means exactly 0 or 1 for classification → overfitting.

- **Best Practices:**

  - Always compute encodings **only on training folds** (use K‑Fold or leave‑one‑out).
  - Add **smoothing** (e.g. Bayesian prior or add‐m) so that rare categories shrink toward the global mean.
  - Store a fallback for unseen categories (global mean).

---

### 3.2 Leave-One-Out Encoding (LOO) <a name="#32-leave-one-out-encoding-loo"></a>

- **Description:**
  For each training row $i$, compute the category mean using all training rows **except** $i$.

  $$
    \text{LOO}(c_i) = \frac{\sum_{j: X_j = c_i} y_j - y_i}{N_{c_i} - 1},
    \quad \text{fallback} \;=\; \mu_{\text{global}}
  $$

- **When to Use:**

  - Small-to-medium training sets where vanilla target encoding overfits easily.
  - When you want a quick “no‑fold” approach to avoid leaking each row’s own target.

- **When _Not_ to Use:**

  - Extremely large datasets (computationally heavier than simple CV).
  - Regression tasks where you need very precise mean estimates (LOO variance can be high).

- **Common Pitfalls:**

  - If $N_c = 1$, you must default to global mean (avoid division by zero).
  - Still leaks category information indirectly (only removes one row, but the remaining $N_c - 1$ rows can still overfit).

- **Best Practices:**

  - Use a global fallback when $N_c=1$.
  - Combine with smoothing: e.g. add a small constant $\alpha$ to numerator/denominator to avoid extreme values for tiny $N_c$.

---

### 3.3 K‑Fold (Out‑of‑Fold) Target Encoding <a name="#33-k‐fold-out-of-fold-target-encoding"></a>

- **Description:**

  1. Split **training set** into $K$ disjoint folds.
  2. For each fold $k$, compute category means on the other $K-1$ folds (leave‑fold‑$k$ out).
  3. Assign those means to fold $k$ → an **out‑of‑fold** encoding for every training row.
  4. For test data, compute a single “global” mean on the full training set, with optional smoothing.

- **When to Use:**

  - Standard approach when doing K‑Fold cross-validation (5‑fold, 10‑fold).
  - Ensures **no leakage** within training data (each row’s encoding never sees its own target).

- **When _Not_ to Use:**

  - If you cannot replicate the exact same folds when training and validating (inconsistent splits cause data leakage risk).
  - If your training set is extremely small (folds may have too few samples per category).

- **Common Pitfalls:**

  - If you use a different random seed or fold splits than your actual model CV, encodings don’t match and you leak.
  - If category appears only in test (or only in a single fold) → fallback required.

- **Best Practices:**

  - Keep a fixed random seed for fold generation.
  - After encoding training, compute a smoothed global mean for test.
  - Combine with a prior (global mean) for rare categories:

    $$
      \text{encoded}(c) = \frac{N_c \cdot \bar{y}_c + m \cdot \bar{y}}{N_c + m}
    $$

    where $m$ is a smoothing parameter, $\bar{y}_c$ is fold‑out category mean, and $\bar{y}$ is global mean.

---

### 3.4 Bayesian‑Smoothed Encoding <a name="#34-bayesian-smoothed-encoding"></a>

- **Description:**
  Add “pseudo‑counts” $\alpha, \beta$ as a Beta($\alpha,\beta$) prior on the category’s positive rate.

  $$
    \text{encoded}(c)
    = \frac{\alpha + \sum_{i: X_i=c} y_i}{\alpha + \beta + N_c}
  $$

  For regression, a conjugate normal‑inverse‑Gamma prior can be used instead.

- **When to Use:**

  - Categories with very few examples (< 5).
  - You need to avoid extreme 0/1 encodings for rare categories.

- **When _Not_ to Use:**

  - If you cannot choose or tune $\alpha,\beta$ appropriately (poor choices over‑shrink or under‑shrink).
  - When using non‑binary targets without an appropriate conjugate prior.

- **Common Pitfalls:**

  - If $\alpha,\beta$ are too small $\to$ little smoothing → rare categories still get extreme values.
  - If $\alpha,\beta$ are too large $\to$ over‑shrink common categories toward global mean.

- **Best Practices:**

  - Choose $\alpha,\beta$ by maximizing likelihood on hold‑out set or via cross‑validation.
  - Use in combination with out‑of‑fold means (K‑Fold target encoding) to avoid leakage.
  - When unseen category appears in test: encode as $\frac{\alpha}{\alpha+\beta}$ (global prior mean).

---

### 3.5 James–Stein (Shrinkage) Encoding <a name="35-james–stein-shrinkage-encoding"></a>

- **Description:**
  A statistical shrinkage method that pulls raw category means $\hat{p}_c$ toward the overall mean $\bar{p}$ with a data‑driven shrinkage factor:

  $$
    \text{JS}(c)
    = \bar{p}
      + \left(1 - \frac{(K-3)\,\bar{p}(1-\bar{p})}{\sum_{d=1}^K N_d(\hat{p}_d - \bar{p})^2}\right)\,(\hat{p}_c - \bar{p}),
  $$

  where $K$ is the number of categories, $N_d$ is sample count, and $\bar{p}$ is global target mean.

- **When to Use:**

  - Many categories with moderate to large counts (> 20 each).
  - You want an “optimal” data‑driven shrinkage without manually selecting priors or smoothing factors.

- **When _Not_ to Use:**

  - Very small datasets: denominator $\sum N_d(\hat{p}_d-\bar{p})^2$ can become unstable.
  - Implementation complexity: if you don’t have a stable library, manual coding is error‑prone.

- **Common Pitfalls:**

  - Numerical instability when $\sum N_d(\hat{p}_d - \bar{p})^2$ is very small.
  - Over‑shrinking when global variance across categories is tiny.

- **Best Practices:**

  - Compute across training data only (no leakage).
  - If denominator is near zero, fall back to global mean $\bar{p}$.
  - After computing shrinkage on train, apply same formula to test (using train statistics).

---

### 3.6 CatBoost‑Style Ordered Target Encoding <a name="36-catboost‑style-ordered-target-encoding"></a>

- **Description:**
  Iterative “online” encoding that never uses a row’s own target or any “future” target values.

  1. Shuffle training rows (or use time order if time‑series).
  2. Maintain running sums and counts for each category.
  3. For each row $i$, encode $X_i=c$ as $\frac{\text{sum}_c}{\text{count}_c}$ from previous rows only.
  4. Update $\text{sum}_c, \text{count}_c$ with $y_i$.

- **When to Use:**

  - When using CatBoost or a similar library that supports it natively.
  - Streaming/online contexts: encode as data arrives in sequence.
  - No need to manually manage K‑Fold splits for leakage avoidance.

- **When _Not_ to Use:**

  - If your data does not have a natural order and you don’t randomize carefully → biases.
  - For small datasets, single random shuffle can still leak “nearly all” target info except the current row.

- **Common Pitfalls:**

  - If you forget to shuffle and data is sorted by label (e.g. all positives first), you understate positive rates early on.
  - Unseen categories at test time → fallback to global mean or prior.

- **Best Practices:**

  - Always shuffle training set (unless time‑dependent).
  - Store running sums/counts from training to encode test/unseen.
  - Use minimal smoothing (e.g. initial count=1 prior) to avoid extreme early estimates.

---

## Choosing the Right Encoder <a name="choosing-the-right-encoder"></a>

Categorical encoding is not one-size-fits-all. The choice depends on:

- **Cardinality:** Number of unique levels in the feature.
- **Model Type:** Whether you use tree-based models, linear models, or neural networks.
- **Dataset Size:** Number of training rows.
- **Predictive Signal:** Whether the category has a strong relationship with the target.

| Scenario                                           | Recommended Encoder(s)                                                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Low‑cardinality (< 20 levels)**                  | One‑Hot Encoding or Label/Ordinal Encoding (trees).                                                                         |
| **Moderate‑cardinality (20–100 levels)**           | One‑Hot (if dimensionality ok) or Frequency Encoding; if strong target signal, use Target Encoding with K‑Fold.             |
| **High‑cardinality (100–1,000 levels)**            | K‑Fold Target Encoding (with smoothing), or Bayesian‑Smoothed Encoding; as fallback, Frequency or Hashing Encoding.         |
| **Extreme‑cardinality (> 1,000 levels)**           | Hashing Encoding (with mean/freq), or Embeddings. If sticking to classical ML, use Bayesian or James–Stein with K‑Fold.     |
| **Tree‑only models (RF, XGBoost, LightGBM)**       | Label/Ordinal Encoding is simplest (trees treat integers as categories).                                                    |
| **Linear/Logistic models (no trees)**              | Supervised encoders (K‑Fold Target, Bayesian‑Smoothed, JS‐Shrink) to capture predictive signal without blowing up features. |
| **Small datasets (< 10k rows)**                    | LOO Encoding or K‑Fold Target Encoding to avoid leakage; add smoothing/prior to stabilize rare categories.                  |
| **Streaming / Online learning**                    | CatBoost “Ordered” Target Encoding or incremental Bayesian encoding (update prior dynamically).                             |
| **Features with time dependency (e.g. date bins)** | Consider ordered encoding keyed by timestamp or explicit time‑aware grouping.                                               |

---

## Key Pitfalls & Best Practices <a name="key-pitfalls--best-practices"></a>

- **Target Leakage:**
  Always compute encodings using only training data. Never use the target from the same row you are encoding. Use K‑Fold or LOO to avoid leakage.
- **Smoothing:**
  Always apply smoothing to avoid extreme values for rare categories. Use Bayesian priors or add pseudo‑counts to shrink rare category means toward the global mean.
- **Fallback for Unseen Categories:**
  Always provide a fallback for unseen categories in test data. Common strategies include using the global mean or a special “unseen” bucket.

- **Leakage Risk:**
  Any supervised encoder that uses the target can leak if you compute category statistics on the same rows you train on. Always use out‑of‑fold or leave‑one‑out schemes.

- **Rare Categories:**
  Categories appearing in only a handful of rows produce unstable means. Always add smoothing (pseudo‑counts or shrinkage) so that rare categories revert toward the global mean.

- **Unseen Categories in Test:**
  Provide a default. Common strategies:

  - Global target mean from training (with or without smoothing).
  - A special “**unseen**” bucket for Frequency or Hashing.

- **Dimensionality vs. Predictive Power:**
  One‑Hot can explode dimensionality for moderate cardinality. Switch to supervised encoders or hashing if dimensionality becomes unmanageable.

- **Interpretability:**
  One‑Hot and Label/Ordinal are highly interpretable. Supervised encoders yield continuous values that can be harder to trace back to specific categories.

- **Regularization:**
  For Bayesian‑smoothed or James–Stein, choose priors that reflect domain knowledge (e.g. rare event base rates). Tune smoothing hyperparameters via cross‑validation when possible.

- **Consistent Splits:**
  If you rely on K‑Fold target encoding, use the exact same folds for your model’s CV procedure. Mismatched folds lead to leakage.

- **Implementation Libraries:**

  - [`category_encoders`](https://contrib.scikit-learn.org/category_encoders/) for Python offers many supervised encoder implementations (Target, Bayesian‑Smoothed, James–Stein, Leave‑One‑Out, etc.), all with built‑in smoothing.
  - Scikit‑Learn’s `HashingEncoder` or custom functions for hashing.
  - Pandas’ `.factorize()` or `sklearn.preprocessing.OrdinalEncoder` for label/ordinal.
  - `sklearn.preprocessing.OneHotEncoder` for one‑hot.

---

## References <a name="references"></a>

This section lists key papers and resources that provide foundational knowledge and practical implementations of the encoders discussed:

1. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. _ACM SIGKDD Explorations Newsletter_, 3(1), 27–32.
2. Rendle, S. (2010). Factorization Machines. _Proceedings of the IEEE International Conference on Data Mining (ICDM)_.
3. Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2018). Target encoding for categorical features. _International Conference on Machine Learning and Applications (ICMLA)_.
4. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. _Advances in Neural Information Processing Systems (NeurIPS)_.

---
