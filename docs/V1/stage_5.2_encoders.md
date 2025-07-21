# 📦 Encoding Strategies for Classical Machine Learning

This document covers encoding techniques—organized by data type—so you can handle:

1. **Categorical variables**
2. **Numerical variables**
3. **Date/Time variables**
4. **Mixed (“hybrid”) variables**

Each section explains **why** and **when** to use a particular encoding, how to implement it, and what pitfalls to watch for. A special focus is placed on **binning** (both supervised and unsupervised), as well as noting any additional real-world data types you might encounter.

---

## Table of Contents

1. [Categorical Encoding](#1-categorical-encoding)
   1.1. [Unsupervised Encoders](#11-unsupervised-encoders)
    • One-Hot Encoder
    • Ordinal Encoder
    • Frequency (“Count”) Encoder
    • Hashing Encoder
   1.2. [Supervised Encoders](#12-supervised-encoders)
    • Target Encoder
    • Weight of Evidence (WOE) Encoder
   1.3. [When to Use / Pitfalls / Tips](#13-when-to-use--pitfalls--tips)

2. [Numerical Encoding](#2-numerical-encoding)
   2.1. [Discretization / Binning](#21-discretization--binning)
    • **Unsupervised Binning**: Equal-Width, Equal-Frequency (Quantile), K-Means
    • **Supervised Binning**: Decision-Tree-Based, Custom Business Logic
   2.2. [Binarization (Thresholding)](#22-binarization-thresholding)
   2.3. [Scaling vs. Encoding](#23-scaling-vs-encoding)
    • Standardization / MinMax / Robust / PowerTransform
   2.4. [When to Use / Pitfalls / Tips](#24-when-to-use--pitfalls--tips)

3. [Date/Time Encoding](#3-datetime-encoding)
   3.1. [Component Extraction](#31-component-extraction)
    • Year / Month / Day / Day of Week / Hour
    • Quarter / Semester / Fiscal Year
    • Decade / Century
   3.2. [Cyclical Encoding](#32-cyclical-encoding)
    • Sine–Cosine Encoding for Cyclic Features
   3.3. [Custom Flags (Day/Night, Weekend, Holidays)](#33-custom-flags-daynight-weekend-holidays)
   3.4. [When to Use / Pitfalls / Tips](#34-when-to-use--pitfalls--tips)

4. [Mixed (“Hybrid”) Variables](#4-mixed-hybrid-variables)
   4.1. [Numerical + Categorical Combination](#41-numerical--categorical-combination)
   4.2. [Multimodal / Text + Numeric + Date](#42-multimodal--text--numeric--date)
   4.3. [When to Use / Pitfalls / Tips](#43-when-to-use--pitfalls--tips)

5. [Additional Real-World Data Types](#5-additional-real-world-data-types)
   5.1. [Boolean / Flag Features](#51-boolean--flag-features)
   5.2. [Geospatial Coordinates](#52-geospatial-coordinates)

6. [Summary Checklist & Best Practices](#6-summary-checklist--best-practices)

---

## 1 — Categorical Encoding<a name="1-categorical-encoding"></a>

Categoricals often appear as strings or enums (e.g., `"State"`, `"Color"`, `"PlanType"`). ML algorithms require numerical inputs, so we must map categories to numbers.

### 1.1 Unsupervised Encoders<a name="11-unsupervised-encoders"></a>

#### 1.1.1 One-Hot Encoder

- **What it does**
  Creates a new binary column for each unique category. If `"Color"` has `{"Red","Blue","Green"}`, we generate three columns:

  ```
  Color_Red  Color_Blue  Color_Green
     1           0            0
     0           1            0
  ```

- **When to use**

  - Nominal categories with relatively small cardinality (< 50).
  - Downstream models: tree-based (e.g. RandomForest) or linear models that handle sparse input.

- **Implementation (scikit-learn)**

  ```python
  from sklearn.preprocessing import OneHotEncoder
  ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
  X_cat = ohe.fit_transform(df[["Color"]])
  ```

- **Pitfalls / Tips**

  - **Cardinality explosion** if too many unique values → memory blowup.
  - **Dummy-variable trap** (multicollinearity) for linear models: drop one column or use `drop="first"`.
  - Use **`handle_unknown="ignore"`** so unseen categories at inference become all-zeros.

#### 1.1.2 Ordinal Encoder

- **What it does**
  Assigns a unique integer to each category arbitrarily (e.g., `{"Low":0,"Medium":1,"High":2}`).
- **When to use**

  - True ordinal categories where order matters (e.g., `Low < Medium < High`).
  - Cardinality is moderate (< 200), and you want a compact numeric representation.

- **Implementation (scikit-learn)**

  ```python
  from sklearn.preprocessing import OrdinalEncoder
  oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
  df["Priority_encoded"] = oe.fit_transform(df[["Priority"]])
  ```

- **Pitfalls / Tips**

  - If categories are not truly ordinal, the numeric ranks impose a spurious order.
  - For unseen categories at inference, set `unknown_value=-1` (or a sentinel).

#### 1.1.3 Frequency (“Count”) Encoder

- **What it does**
  Replaces each category by its relative frequency (or raw count) in the training set.
  E.g., if `"City=Seattle"` appears 12 % of times → encode as 0.12.
- **When to use**

  - High-cardinality nominal features (> 100).
  - You want a quick continuous proxy of category importance without expanding dimension.

- **Implementation**

  ```python
  freq_map = df["City"].value_counts(normalize=True).to_dict()
  df["City_freq"] = df["City"].map(freq_map).fillna(0.0)
  ```

- **Pitfalls / Tips**

  - **Leakage risk**: compute frequencies on **train only**, then apply to test.
  - Rare categories (< 1 %) → very small freq values; consider lumping into a `__rare__` bucket.

#### 1.1.4 Hashing Encoder

- **What it does**
  Hashes each category string into a fixed number of integer “buckets” (dimensions).
  E.g., with `n_components=16`, `"California"` might hash to bucket 3, `"Texas"` to bucket 11.
- **When to use**

  - Extremely high-cardinality features (thousands of levels).
  - Memory/time constraint: hashing avoids storing large dictionaries.

- **Implementation (`category_encoders` library)**

  ```python
  from category_encoders.hashing import HashingEncoder
  he = HashingEncoder(n_components=16, drop_invariant=True)
  df_hash = he.fit_transform(df[["ZipCode"]])
  ```

- **Pitfalls / Tips**

  - **Hash collisions**: distinct categories can map to the same bucket → noisy signal.
  - Choose `n_components` large enough (e.g., `n_components ≈ #unique_cats / 10`).
  - No need to track a dictionary—beneficial in streaming/incremental settings.

---

### 1.2 Supervised Encoders<a name="12-supervised-encoders"></a>

#### 1.2.1 Target Encoder

- **What it does**
  Replaces each category by the _average_ of the target variable in the training set.
  For classification, `mean(target | category)`; for regression, `mean(y)` for that category.
- **When to use**

  - High-cardinality nominal features where one-hot → too many columns.
  - You expect a strong correlation between category and target.

- **Implementation (`category_encoders` library)**

  ```python
  from category_encoders import TargetEncoder
  te = TargetEncoder(smoothing=0.3)
  X_te = te.fit_transform(df[["ZipCode"]], df["Churn"])
  ```

  > **Smoothing** prevents overfitting on rare categories by shrinking small‐group means toward the global mean.

- **Pitfalls / Tips**

  - **Leakage risk**: must compute encoding on **train** only (or via cross-fold) then apply to validation/test.
  - Use **cross-fold mean** or **leave-one-out** in training to reduce bias.
  - Rare categories produce noisy estimates; apply smoothing or minimum‐count thresholds.

#### 1.2.2 Weight of Evidence (WOE) Encoder

- **What it does**
  For binary classification, `WOE = ln( (Good%)/(Bad%) )` for each category (`Good`/`Bad` = target classes).
  Creates a continuous column capturing each category’s log-odds.
- **When to use**

  - Credit-scoring or risk-modeling domains.
  - When interpretability is key: WOE values align linearly in logistic regression.

- **Implementation (`category_encoders` library)**

  ```python
  from category_encoders import WOEEncoder
  woe = WOEEncoder()
  df_woe = woe.fit_transform(df[["Occupation"]], df["Defaulted"])
  ```

- **Pitfalls / Tips**

  - Not applicable if target is continuous or multi-class (designed for binary).
  - Rare categories with zero events → infinite WOE. Must apply **smoothing** or cap values.
  - As with target encoder, avoid leakage by calculating WOE only on training folds.

---

### 1.3 When to Use / Pitfalls / Tips<a name="13-when-to-use--pitfalls--tips"></a>

| Encoder Type  | Use When …                                                         | Watch Out For …                                                     |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| **One-Hot**   | Few unique categories (< 50); algorithms benefit from sparse input | Dimensionality explosion if cardinality too high; multicollinearity |
| **Ordinal**   | True ordinal features (`Low < Medium < High`)                      | Imposing false order on nominal data                                |
| **Frequency** | Moderate-to-high cardinality (50–500); want continuous proxy       | Train/test distribution shift; rare categories get tiny freq        |
| **Hashing**   | Extremely high cardinality (> 500); memory/time constraints        | Collisions introduce noise                                          |
| **Target**    | Strong category→target relationship; need compact representation   | Data leakage—fit on train only; smoothing needed                    |
| **WOE**       | Credit-scoring; binary target; interpretability matters            | Infinite WOE for zero counts; smoothing; strictly binary only       |

- **Always** fit supervised encoders **only** on the training set (or within cross-validation folds) to avoid leakage.
- When cardinality is high (> 100), consider Frequency or Hashing first; if you suspect a strong signal, use Target or WOE with care.
- For **rare categories** (< 1 %), group into a single `__rare__` bucket before encoding.

---

## 2 — Numerical Encoding<a name="2-numerical-encoding"></a>

Numerical encoding turns continuous or numeric data into discrete bins or binary indicators, or otherwise transforms its scale for better model performance. A special emphasis is on **binning** with both **unsupervised** and **supervised** approaches.

### 2.1 Discretization / Binning<a name="21-discretization--binning"></a>

#### 2.1.1 Unsupervised Binning

These methods ignore the target variable and split purely on the distribution of the feature itself.

- **Equal-Width Binning**

  - **What it does**: Splits the numeric range into _k_ equal-sized intervals.
    E.g., if `x` ranges \[0..100] and `k=5`, bins are `[0–20),[20–40),…,[80–100]`.
  - **When to use**:
    • Numeric is roughly uniform; simple buckets suffice.
    • Outliers are few—outlier effects may be diluted if bins are wide.
  - **Implementation (scikit-learn)**

    ```python
    from sklearn.preprocessing import KBinsDiscretizer
    ew = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    df["age_ewbin"] = ew.fit_transform(df[["age"]])
    ```

  - **Pitfalls / Tips**:
    • Skewed data → bins have very different counts.
    • Choose `k` by visually inspecting histograms first.

- **Equal-Frequency (Quantile) Binning**

  - **What it does**: Splits into _k_ bins such that each bin has (roughly) the same number of samples.
  - **When to use**:
    • Highly skewed distributions; ensures balanced representation across bins.
  - **Implementation (scikit-learn)**

    ```python
    qf = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    df["income_qbin"] = qf.fit_transform(df[["income"]])
    ```

  - **Pitfalls / Tips**:
    • Bin boundaries can collapse if many tied values.
    • Interpretability of bin boundaries may be less intuitive than equal-width.

- **K-Means (Clustering) Binning**

  - **What it does**: Uses 1D K-means clustering on the feature to find _k_ cluster centers, then assigns bin labels accordingly.
  - **When to use**:
    • Suspect natural “clusters” in that feature (e.g., spending levels).
  - **Implementation**

    ```python
    from sklearn.cluster import KMeans
    arr = df["transaction_amount"].values.reshape(-1,1)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(arr)
    df["amt_cluster"] = kmeans.labels_
    ```

  - **Pitfalls / Tips**:
    • More compute‐intensive than uniform/quantile binning.
    • Results depend on initialization—set `random_state` for reproducibility.
    • May produce non‐contiguous bins if values intersperse—inspect cluster centers.

#### 2.1.2 Supervised Binning

These methods use the target variable to form bins that maximize predictive power.

- **Decision-Tree-Based Binning**

  - **What it does**: Fits a shallow decision tree (e.g., `max_leaf_nodes=k` or `max_depth=d`) on one numeric feature vs. target and uses the tree splits as bin thresholds.
  - **When to use**:
    • You want bins that directly optimize separation by the target.
    • Good for binary classification—tree finds thresholds where target probability changes significantly.
  - **Implementation (example)**

    ```python
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    # For classification, max_leaf_nodes = desired # of bins
    dt = DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=0.05, random_state=0)
    # Fit on numeric column and target
    dt.fit(df[["feature"]], df["target"])
    # Extract thresholds from tree
    thresholds = sorted(set(
        dt.tree_.threshold[dt.tree_.threshold != -2]  # −2 is leaf indicator
    ))
    # Use pd.cut to bin
    df["feature_dtbin"] = pd.cut(df["feature"], bins=[-np.inf] + thresholds + [np.inf], labels=False)
    ```

  - **Pitfalls / Tips**:
    • May overfit if tree is too deep—limit `max_leaf_nodes` or `max_depth`.
    • For regression, use `DecisionTreeRegressor` and look at splits.
    • Bins optimal for train may not generalize; validate on held-out data.

- **Custom Business-Logic Binning**

  - **What it does**: You define thresholds based on domain knowledge.
    E.g., “Age groups: 0–17\*\*, 18–34\*\*, 35–54\*\*, 55+” for marketing campaigns.
  - **When to use**:
    • When you know meaningful intervals—regulatory categories, pricing tiers, age brackets, etc.
    • Ensures interpretability (“we know why 18–34 is separate”).
  - **Implementation**

    ```python
    bins = [0, 18, 35, 55, 120]
    labels = ["0–17", "18–34", "35–54", "55+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    ```

  - **Pitfalls / Tips**:
    • Requires strong domain knowledge; thresholds may need regular updates if business context changes.
    • Always document reasoning in a **Feature Dictionary**.

---

### 2.2 Binarization (Thresholding)<a name="22-binarization-thresholding"></a>

- **What it does**
  Converts a numeric feature into binary (0/1) by threshold:

  ```python
  df["high_income"] = (df["income"] > 50_000).astype(int)
  ```

- **When to use**

  - If you know a meaningful cut-off (e.g., “Age ≥ 65 vs. < 65”).
  - When you want a simple indicator rather than a continuous variable.

- **Implementation (scikit-learn)**

  ```python
  from sklearn.preprocessing import Binarizer
  binarizer = Binarizer(threshold=1000.0)  # values > 1000 → 1
  df["payment_flag"] = binarizer.fit_transform(df[["monthly_payment"]])
  ```

- **Pitfalls / Tips**

  - **Choosing threshold:** domain knowledge is critical.
  - Binarization loses information if over-applied; use sparingly.

---

### 2.3 Scaling vs. Encoding<a name="23-scaling-vs-encoding"></a>

Often conflated with “encoding,” **scaling** adjusts the numeric range—`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PowerTransformer`. Note:

- **Scaling** does _not_ discretize; it rescales continuous features for gradient-based or distance-based algorithms.
- **Encoding** (discretization, binarization) transforms numeric to categorical-like codes.

| Transformer             | Effect                                                        | `sklearn` Class                                     |
| ----------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| **StandardScaler**      | Zero mean, unit variance                                      | `StandardScaler()`                                  |
| **MinMaxScaler**        | Scales feature to \[0,1]                                      | `MinMaxScaler()`                                    |
| **RobustScaler**        | Scales via IQR (median-based); robust to outliers             | `RobustScaler()`                                    |
| **PowerTransformer**    | Yeo–Johnson / Box–Cox (makes distribution more Gaussian-like) | `PowerTransformer(method="yeo-johnson")`            |
| **QuantileTransformer** | Maps to uniform or normal distribution (rank-based)           | `QuantileTransformer(output_distribution="normal")` |

**Pitfalls / Tips**

- If numeric feature is highly skewed, consider `PowerTransformer` **before** or **instead of** binning.
- When combining binning + scaling, decide if you want to scale the original or the binned indices.

---

### 2.4 When to Use / Pitfalls / Tips<a name="24-when-to-use--pitfalls--tips"></a>

| Encoding Type               | Use When …                                                                    | Pitfalls / Tips                                                                    |
| --------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Equal-Width Binning**     | Numeric is roughly uniform; bins of equal width make sense                    | Skewed data → bins have imbalanced counts; may obscure patterns                    |
| **Equal-Frequency Binning** | Skewed data; want each bin to have equal samples                              | Bin intervals may be unintuitive; many tied values → empty bins                    |
| **K-Means Binning**         | Suspect natural clusters in that single feature                               | More compute; cluster centers can shift with new data—incompatibility over time    |
| **Decision-Tree Binning**   | Want bins that directly optimize separation by target (supervised)            | May overfit; tree splits vary if data distribution shifts; validate on unseen data |
| **Custom Business-Logic**   | Domain knowledge defines meaningful thresholds (e.g., age brackets)           | Must maintain thresholds as business changes; document rationale                   |
| **Binarization**            | Known cutoff yields simple indicator (e.g., “age ≥ 65”)                       | Loses information; threshold must be well-justified                                |
| **Scaling/PowerTransform**  | Many ML algorithms (SVM, KNN, Logistic Regression) need roughly normal inputs | If feature already well-distributed, transform may be unnecessary or harmful       |

---

## 3 — Date/Time Encoding<a name="3-datetime-encoding"></a>

Dates and timestamps carry rich information. We can decompose into calendar, cyclical signals, or custom flags.

### 3.1 Component Extraction<a name="31-component-extraction"></a>

Extract discrete fields from a timestamp:

- **Year**: integer (e.g., 2023).
- **Month**: integer 1–12 (or one-hot for clear seasonality).
- **Day of Month**: 1–31 (rarely used standalone).
- **Day of Week**: 0 (Monday) … 6 (Sunday).
- **Hour**: 0–23 (for intra-day patterns).
- **Quarter**: 1–4 (fiscal/business quarter).
- **Semester**: 1 or 2 (academic year half).
- **Decade**: `(year // 10) * 10` (e.g., 1995→1990).
- **Century**: `(year // 100) * 100` (rare in typical ML).

```python
df["event_time"] = pd.to_datetime(df["event_time"])
df["year"] = df["event_time"].dt.year
df["month"] = df["event_time"].dt.month
df["day"] = df["event_time"].dt.day
df["day_of_week"] = df["event_time"].dt.dayofweek  # 0=Mon … 6=Sun
df["hour"] = df["event_time"].dt.hour
df["quarter"] = df["event_time"].dt.quarter
df["semester"] = df["event_time"].dt.month.apply(lambda m: 1 if m <= 6 else 2)
df["decade"] = (df["year"] // 10) * 10
df["century"] = (df["year"] // 100) * 100
```

- **When to Extract**:
  • Early in pipeline, after reading raw data; extract from main “timestamp” column, then drop the original if not needed.
- **Pitfalls / Tips**:

  - If you use `month` or `hour` as plain integers, ML models treat them linearly. Use **cyclical encoding** to preserve adjacency.

---

### 3.2 Cyclical Encoding<a name="32-cyclical-encoding"></a>

To preserve circular continuity (e.g., December→January, Hour 23→Hour 0), map:

$$
x_{\sin} = \sin\!\Bigl(\frac{2\pi\,x}{\text{period}}\Bigr),\quad
x_{\cos} = \cos\!\Bigl(\frac{2\pi\,x}{\text{period}}\Bigr).
$$

- **Examples**:

  ```python
  # Month (1–12)
  df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
  df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

  # Hour (0–23)
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

  # Day of week (0–6)
  df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
  df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
  ```

- **When to use**

  - Any cyclic feature (hour, month, day-of-week, etc.) that would be mis-ordered by plain ordinal encoding.

- **Pitfalls / Tips**

  - Drop the original integer column once sine/cosine are constructed (two inputs suffice).
  - Wrap in a `ColumnTransformer` via `FunctionTransformer` for clarity.

---

### 3.3 Custom Flags (Day/Night, Weekend, Holidays)<a name="33-custom-flags-daynight-weekend-holidays"></a>

#### Day vs Night

- **Use case**: user activity differs drastically at night vs daytime.

  ```python
  df["is_day"] = df["hour"].apply(lambda h: 1 if 6 <= h < 18 else 0)
  ```

#### Weekend vs Weekday

- **Use case**: business metrics differ on weekends.

  ```python
  df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
  ```

#### Public Holidays (Country-Specific)

- **Use case**: traffic spikes on holidays.
  Use package `holidays` to cross-check:

  ```python
  import holidays
  us_holidays = holidays.CountryHoliday("US")
  df["is_holiday"] = df["event_time"].dt.date.apply(lambda d: int(d in us_holidays))
  ```

- **Pitfalls / Tips**

  - Holiday calendars differ by country/time zone—select correct locale.
  - If “pre-holiday” or “post-holiday” effects matter, create additional flags accordingly.

---

### 3.4 When to Use / Pitfalls / Tips<a name="34-when-to-use--pitfalls--tips"></a>

| Encoding Type                 | Use When …                                                                        | Pitfalls / Tips                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Year / Quarter / Month**    | Long-term trends (year), seasonal cycles (quarter/month)                          | If strictly linear, treats Jan→Feb as small step but Dec→Jan as large gap—use cyclical instead.  |
| **Cyclical (sin/cos)**        | Hour, month, day-of-week, or any repeating cycle                                  | After cyclical encoding, drop original integer column (redundant).                               |
| **Custom Day/Night, Weekend** | Domain suggests strong day/night or weekend vs weekday behavior                   | Check time zones carefully; if time zone varies across data, convert to a common zone first.     |
| **Holiday Flags**             | Domain-specific events affecting user behavior (bank holidays, national holidays) | Maintaining holiday lists is extra work; rely on a maintained library for your region/time zone. |

---

## 4 — Mixed (“Hybrid”) Variables<a name="4-mixed-hybrid-variables"></a>

A “mixed” variable might combine numeric, categorical, or date semantics, or you may want to engineer features that cross-combine multiple types.

### 4.1 Numerical + Categorical Combination<a name="41-numerical--categorical-combination"></a>

- **Example**: “Days Since Last Purchase” (numeric) combined with “Membership Tier” (categorical) to create an “Urgency Score.”

  ```python
  df["days_since"] = (pd.Timestamp("now") - df["last_purchase"]).dt.days
  tier_map = {"Bronze": 0.5, "Silver": 1.0, "Gold": 1.5}
  df["tier_score"] = df["membership_tier"].map(tier_map)
  df["urgency_score"] = df["days_since"] / df["tier_score"]
  ```

- **When to use**

  - Domain knowledge suggests direct interplay between a numeric and a category.
  - Interaction features help linear models capture non-linear effects.

- **Pitfalls / Tips**

  - Engineered features can be dataset-specific—document them carefully.
  - Avoid dividing by zero or missing denominators—impute or add small epsilon.

### 4.2 Multimodal / Text + Numeric + Date<a name="42-multimodal--text--numeric--date"></a>

If your table has text, numeric, date/time all together, build separate pipelines and concatenate:

1. **Numeric pipeline**

   - Impute → Scale/PowerTransform → (Optionally) Bin or Interaction.

2. **Categorical pipeline**

   - Rare-bucket → One-Hot/Target/WOE → (Optionally) Combine with numeric.

3. **Text pipeline**

   - Create length/count features → TF-IDF vector → (Optionally) topic modeling.

4. **Date/Time pipeline**

   - Extract year/month/day/etc. → Cyclical sin/cos → Holiday/Weekend flags.

Example (scikit-learn `ColumnTransformer` skeleton):

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

numeric_features = ["age", "income"]
categorical_features = ["City", "PlanType"]
text_features = ["ReviewText"]
date_features = ["SignupDate"]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("rare", FunctionTransformer(lambda df: df.mask(df.map(df.value_counts(normalize=True)) < 0.01, "__rare__"))),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

text_pipeline = Pipeline([
    ("lengths", FunctionTransformer(lambda df: pd.DataFrame({
        f"{text}__n_chars": df[text].fillna("").str.len(),
        f"{text}__n_words": df[text].fillna("").str.split().str.len(),
    }))),
    ("tfidf", TfidfVectorizer(max_features=100)),
])

date_pipeline = Pipeline([
    ("extract", FunctionTransformer(lambda df: pd.DataFrame({
        "year": df["SignupDate"].dt.year,
        "month_sin": np.sin(2*np.pi*df["SignupDate"].dt.month/12),
        "month_cos": np.cos(2*np.pi*df["SignupDate"].dt.month/12),
        "is_weekend": df["SignupDate"].dt.dayofweek.isin([5,6]).astype(int),
    }))),
    ("scale", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
    ("txt", text_pipeline, text_features),
    ("date", date_pipeline, date_features),
], remainder="drop")

# Then:
# X_transformed = preprocessor.fit_transform(df)
```

### 4.3 When to Use / Pitfalls / Tips<a name="43-when-to-use--pitfalls--tips"></a>

| Scenario                                              | Suggested Action                                                           | Pitfalls / Tips                                                                                                                  |
| ----------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Numeric feature interacts with a category             | Create derived column (e.g. “amount × category_score”)                     | Compute on train only; document in a **Feature Dictionary**.                                                                     |
| Multiple modalities (text + numeric + date)           | Build separate pipelines per modality and **concatenate**                  | Ensure pipeline handles missing values; check that text vocabulary stays within memory.                                          |
| “Mixed” column (e.g. `"A=1;B=2;C=text;D=2021-01-01"`) | First split into individual fields, parse types, then encode each subfield | Use robust `pandas` string parsing or regex, avoid ad-hoc splits; treat each sub-field carefully (impute missing, correct type). |

---

## 5 — Additional Real-World Data Types<a name="5-additional-real-world-data-types"></a>

Beyond purely categorical, numeric, and date, real-world tabular datasets may include:

### 5.1 Boolean / Flag Features<a name="51-boolean--flag-features"></a>

- **What it is**: columns with only two values (e.g., `True/False`, `0/1`, `"Y"/"N"`).
- **How to encode**:

  - Map directly to `0/1` integers.

    ```python
    df["has_discount"] = df["has_discount"].map({True: 1, False: 0})
    ```

  - Or leave as boolean if downstream pipeline accepts booleans (many libraries internally cast to `0/1`).

- **When to use**:

  - Very straightforward; no dimensional explosion.

- **Pitfalls / Tips**:

  - Watch for inconsistent representations: `"Y"/"N"`, `"yes"/"no"`, `True/False`, `1/0`. Normalize early.
  - Ensure no missing values—impute to `0` or `False` if appropriate.

### 5.2 Geospatial Coordinates<a name="52-geospatial-coordinates"></a>

- **What it is**: latitude/longitude pairs, postal codes, address strings.
- **How to encode**:

  - **Latitude / Longitude**: leave as two numeric columns if spatial model can consume.
  - **Clustering / Bucketing**: Cluster GPS points into regions (e.g., KMeans on lat/lon) → region ID.
  - **Haversine Distance**: If you have a “central” point (office location), compute Haversine distance as a numeric feature.

    ```python
    import numpy as np

    def haversine(lat1, lon1, lat2, lon2):
        # approximation in kilometers
        R = 6371.0
        φ1, φ2 = np.radians(lat1), np.radians(lat2)
        Δφ = φ2 - φ1
        Δλ = np.radians(lon2 - lon1)
        a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df["dist_to_center_km"] = haversine(
        df["lat"], df["lon"], center_lat, center_lon
    )
    ```

- **When to use**:

  - Spatial clustering (e.g., market segmentation by region).
  - Distance features for location-based predictions (e.g., delivery time, travel cost).

- **Pitfalls / Tips**:

  - Latitude/longitude wrap around at ±180° longitude and ±90° latitude—ensure correct range.
  - If addresses are strings, geocoding (external API) may be required—be mindful of rate limits.

---

## 6 — Summary Checklist & Best Practices<a name="6-summary-checklist--best-practices"></a>

1. **Identify Data Types First**

   - Numeric → decide on scaling vs discretization vs binarization.
   - Categorical → pick one-hot/ordinal/frequency/hash/target/WOE.
   - Date/Time → extract components, then cyclical encode.
   - Text → (if present) length/Tf-IDF, not covered here in depth.
   - Boolean → map to 0/1.
   - Geospatial → either numeric lat/lon, cluster, or distance features.

2. **Cardinality Considerations**

   - _Low cardinality_ (<50): one-hot (sparingly).
   - _Medium cardinality_ (50–500): ordinal or frequency encoder.
   - _High cardinality_ (>500): hashing or supervised encoders (target, WOE with caution).

3. **Supervised vs. Unsupervised Binning**

   - _Unsupervised_: equal-width, equal-frequency, K-means—use when ignoring target is fine.
   - _Supervised_: decision-tree splits or custom business rules—use when you know target grouping helps separation.

4. **Avoid Leakage**

   - Fit all supervised encoders (Target, WOE, decision-tree binning) **only** on the training set.
   - Validate on hold-out data to ensure bins/generalized encoding remain stable.

5. **Handle Rare/Unseen Categories**

   - Categories < 1 % → lump into a `__rare__` bucket before encoding.
   - For unseen test categories, `OneHotEncoder(handle_unknown="ignore")` or `OrdinalEncoder(unknown_value=-1)`.

6. **Cyclical Variables**

   - Use sin/cos for Hour, Month, Day-of-Week, etc., to preserve neighbor relationships.
   - Drop original integer column after cyclical encoding.

7. **Discretization vs. Continuous**

   - Discretize only when intervals are meaningful (e.g., business tiers).
   - Otherwise, scaling (Standard, MinMax, PowerTransform) often suffices.

8. **Document Everything**

   - Maintain a **Feature Dictionary** (e.g., `docs/feature_dictionary.md`) listing each engineered column, its transformation, and rationale.
   - Include before/after distribution plots (histogram, boxplot) in EDA reports.

9. **Test Pipelines End-to-End**

   - Build a `ColumnTransformer` or `Pipeline` orchestrating all steps.
   - Ensure `transform()` on unseen data does not error out (especially for unseen categories).
   - Unit-test custom encoders on edge cases (all zeros, all missing, unseen categories).

10. **Iterate & Validate**

- Validate engineered features via **feature importance** (permutation importance, mutual information).
- If a feature has very low signal, consider dropping to reduce noise.

11. **Document Business Logic**

- Any **custom binning** (e.g., age brackets, pricing tiers) must be explicitly documented.
- If thresholds change over time (e.g., age groups adjusted for new regulations), update documentation immediately.

12. **Additional Data Types**

- **Boolean** → direct 0/1 or keep as boolean if pipeline accepts.
- **Geospatial** → lat/lon (numeric), distance to point, region clustering.
- **Text** → length, Tf-IDF (out of scope for classical tabular focus).
- **Mixed fields** → split into subfields, encode each appropriately, then combine.

---

> **Bottom Line**:
>
> - **Categorical** → choose one-hot/ordinal/frequency/hashing/target/WOE based on cardinality & supervision.
> - **Numerical** → use scaling or discretizing (equal-width, quantile, K-means, decision-tree, custom).
> - **Date/Time** → decompose into components, then cyclical encode.
> - **Mixed Variables** → split by type, encode individually, and combine as needed.

Use this guide as a **living reference**—update whenever you adopt a new encoder or discover a novel transformation.
