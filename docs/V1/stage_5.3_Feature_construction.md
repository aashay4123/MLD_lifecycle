# Feature Construction & Splitting

_A Practical Methodology for Building Strong Features and Sound Data Splits_

---

# 1. Overview

Feature construction and data splitting are two pillars of a robust ML pipeline.

- **Feature Construction ** turns raw columns into expressive, predictive inputs.
- **Data Splitting ** partitions your dataset so that evaluation is honest and leakage-free.

This guide walks you through proven patterns, tips, and pitfalls for each.

---

# 2. Feature Construction

# 2.1. Guiding Principles

1. ** Domain Knowledge First**

- Leverage business logic, physics, or biology to craft features.
- Example: In bioinformatics, enzyme kinetics → Michaelis–Menten ratio.

2. ** Keep It Simple(to start)**

- Start with linear or logarithmic transforms before jumping to complex encodings.
- Validate each new feature with a quick correlation or importance check.

3. ** Iterate & Validate**

- Build one family of features at a time(e.g. temporal, aggregations, interactions).
- Use cross-validated performance or mutual information to prune.

4. ** Avoid Data Leakage**

- Never use future information(e.g. post-event aggregates) when constructing training features.

---

# 2.2. Common Patterns

| Category               | Techniques                                     | When to Use                                        |
| ---------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **Transforms **        | log, √, Box-Cox, Yeo-Johnson                   | Skewed numeric, heavy tails                        |
| **Scaling **           | Standard, MinMax, Robust                       | Before distance-based models(KNN, SVM, clustering) |
| **Binning & Discret.** | Quantile, uniform, k-means, decision-tree bins | Non-linear models, interpretability needs          |
| **Interactions **      | Pairwise products, ratios                      | Suspected synergy between features                 |
| **Polynomial **        | Degree 2–3                                     | Smooth non-linearities(with caution)               |
| **Aggregations **      | Group-by mean/sum/std/min/max, rolling windows | Hierarchical or time-series data                   |
| **Datetime **          | Year/Month/Day/DOW/HOD, cyclic sin/cos         | Seasonality, periodic patterns                     |
| **Categorical **       | Target, frequency, embeddings, hashing         | High-cardinality, tree vs. linear model decisions  |
| **Text **              | TF-IDF, word counts, embeddings                | NLP features for reviews or notes                  |
| **Dimensionality **    | PCA, TSVD, autoencoders                        | Collinearity reduction, visualization              |

---

# 2.3. Workflow Example

1. ** Inspect raw distributions ** → identify skew, zeros, outliers.
2. ** Basic transforms ** → log1p / Box-Cox on skewed columns.
3. ** Create domain features ** → ratios, differences, polynomial terms.
4. ** Aggregate ** → group-level statistics(e.g. per-patient averages).
5. ** Encode categoricals ** → based on cardinality & model choice.
6. ** Test feature utility ** → quick univariate mutual information or importance.
7. ** Select & prune ** → remove near-zero variance or highly correlated features.

---

# 3. Data Splitting

# 3.1. Core Goals

- **Generalization**: simulate unseen data
- **Fair Evaluation**: no leakage between train/test
- **Representativeness**: preserve class balance or time continuity

# 3.2. Strategies

| Split Type           | Method                             | Use Case                                     |
| -------------------- | ---------------------------------- | -------------------------------------------- |
| **Random Hold-out ** | `train_test_split(stratify=…)`     | Baseline, balanced classification            |
| **K-Fold CV **       | StratifiedShuffleSplit / KFold     | Small datasets, variance estimation          |
| **Time-Series **     | Expanding window / rolling forward | Forecasting, temporal autocorrelation        |
| **Grouped **         | GroupKFold                         | Avoid leakage when multiple rows share an ID |
| **Stratified **      | preserve target distribution       | Imbalanced classes, rare events              |

# 3.3. Best Practices

1. ** Stratify on the Target**

- For classification, preserve class ratios in train & test.

2. ** Group-Aware Splits**

- If one entity appears in multiple rows(e.g. patient, customer), group-split to avoid leakage.

3. ** Time-Aware Splits**

- Ensure test periods always occur after train periods.

4. ** Validation Set**

- Hold out both a ** validation ** and a ** final test ** to tune hyperparameters and then measure final performance.

5. ** Reproducibility**

- Fix `random_state` or seeds across splits.

---

# 4. Putting It All Together

1. ** Plan your split ** based on data type: random vs. grouped vs. temporal.
2. ** Construct features on TRAIN ONLY**:
   - Fit imputers, scalers, encoders on train.
   - Apply identical transformations to validation/test.
3. ** Evaluate feature candidates ** via cross-validation within train.
4. ** Finalize**:
   - Freeze your feature pipeline.
   - Apply once, export processed datasets.

```mermaid
flowchart TD
A[Raw Data] - -> B[Train/Test Split]
B - -> C[Train Data]
C - -> D[Missing-Value Impute]
D - -> E[Outlier Detection & Removal]
E - -> F[Scaling & Transform]
F - -> G[Feature Construction]
G - -> H[Feature Selection]
H - -> I[Encoded Train Ready]
B - -> J[Test Data]
J - -> | Apply same pipeline | I[Test Ready]
```

---

# 5. Tips & Pitfalls

- **Pitfall**: Using test data to choose features → severe over-optimism.
- **Tip**: Log distributions before/after each step for sanity checks.
- **Tip**: Use mutual information or permutation importance to validate new features.
- **Pitfall**: High-cardinality categorical one-hot → explosion of columns.
- **Tip**: When in doubt, start with frequency or hashing encodings.

---

_Keep this guide on hand as a checklist while you engineer and split your features—your future self will thank you!_
