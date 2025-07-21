## Outlier Detection & Treatment

Outliers—extreme or anomalous data points—can arise for many reasons: measurement errors, data entry mistakes, technical artifacts, or genuine but rare biological events. Properly identifying and handling outliers is crucial because they can:

- **Skew summary statistics** (e.g., mean, variance)
- **Distort model estimates** (especially linear models)
- **Create misleading correlations**
- **Trigger false positives/negatives** in downstream analyses

Below is a dedicated, stand‑alone guide to outlier detection and treatment. It covers **when** to address outliers (and when not to), and details the common **univariate**, **bivariate**, and **multivariate** methods you can apply—along with their pros, cons, and caveats.

---

### Table of Contents

1. [When to Detect & Treat Outliers](#when-to-detect--treat-outliers)
2. [When **Not** to Treat Outliers](#when-not-to-treat-outliers)
3. [Univariate Outlier Methods](#univariate-outlier-methods)

   - [IQR (Boxplot) Method](#iqr-boxplot-method)
   - [Z‑Score Method](#z‑score-method)
   - [Modified Z‑Score (Median Absolute Deviation)](#modified-z‑score-median-absolute-deviation)
   - [Percentile‑Based Cutoffs](#percentile‑based-cutoffs)

4. [Bivariate & Pairwise Outlier Analysis](#bivariate--pairwise-outlier-analysis)

   - [Scatterplot Inspect & Visual Thresholds](#scatterplot-inspect--visual-thresholds)
   - [Mahalanobis Distance in 2D](#mahalanobis-distance-in-2d)
   - [Regression‑Based Residuals](#regression‑based-residuals)

5. [Multivariate Outlier Methods](#multivariate-outlier-methods)

   - [Mahalanobis Distance (Generalized)](#mahalanobis-distance-generalized)
   - [Principal Component or Robust PCA Methods](#principal-component-or-robust-pca-methods)
   - [Isolation Forest](#isolation-forest)
   - [Local Outlier Factor (LOF)](#local-outlier-factor-lof)
   - [One‑Class SVM](#one‑class-svm)
   - [Clustering‑Based Approaches (e.g., DBSCAN)](#clustering‑based-approaches-eg-dbscan)

6. [Choosing the Right Method](#choosing-the-right-method)
7. [Treatment Strategies](#treatment-strategies)

   - [Removal vs. Imputation](#removal-vs-imputation)
   - [Transformation (Winsorizing, Truncation)](#transformation-winsorizing-truncation)
   - [Indicator Flags & Robust Models](#indicator-flags--robust-models)

8. [Common Pitfalls & Best Practices](#common-pitfalls--best-practices)
9. [Summary Checklist](#summary-checklist)

---

## When to Detect & Treat Outliers

1. **Prevent Distorted Statistical Summaries**

   - Example: A single sample with extremely high “gene count” can inflate the mean, misleading downstream thresholds (e.g., differential expression).
   - If you plan to use methods sensitive to means/variances (e.g., linear regression, PCA), unaddressed extreme values can dominate results.

2. **Ensure Model Robustness**

   - Many classical ML models (e.g., linear/logistic regression, k‑means clustering) assume roughly symmetric, light‑tailed distributions. Outliers can pull coefficients or cluster centroids far from the “bulk” of data.
   - In bioinformatics, for instance, one spurious protein expression in a small cohort can lead to an erroneous biomarker.

3. **Improve Visualization & Interpretation**

   - A handful of extreme points can obscure visual patterns (e.g., boxplots become unreadable, scatterplots compressed).
   - Early detection helps focus on the main data distribution.

4. **Detect Technical Errors**

   - If a measurement device malfunctioned for certain samples, outliers are the red flags.
   - Example: A QC failure in one RNA‑Seq lane might produce abnormally high read counts for a few genes.

5. **Highlight Biological Rarities**

   - Rare but genuine subpopulations (e.g., a subgroup of cells with extreme marker expression) should be identified as potential signals, not necessarily removed.
   - Distinguishing true biological outliers from technical artifacts is key.

6. **Standardize Preprocessing**

   - Adding an explicit outlier‑detection step ensures reproducible pipelines—every user knows how and where outliers were handled.

---

## When **Not** to Treat Outliers

1. **Genuine Biological Extremes**

   - If the outlier reflects a real biology (e.g., ultra‑high expression in a tumor subtype), removing it may discard the very signal you wish to study.
   - Always cross‑check: Is that outlier associated with a known subgroup (e.g., rare disease mutation)? If so, treat it as a special class rather than noise.

2. **Small Sample Sizes & Data Sparsity**

   - In tiny datasets (e.g., n = 20), a few extreme points might be meaningful—statistical methods to detect outliers may be underpowered or misleading.
   - Over‑zealous removal can exacerbate data scarcity.

3. **When Using Robust Methods by Design**

   - Some algorithms (e.g., tree‑based ensembles, random forests) are inherently robust to outliers. In such cases, explicit outlier removal might not yield significant gains.
   - However, if poor data quality (e.g., input error) rather than extreme but valid values is the issue, you may still want to correct or remove.

4. **When Downstream Models Are Robust**

   - Techniques like **Huber regression** or **quantile regression** are less sensitive to extreme residuals. If the modeling choice already accounts for heavy tails, consider skipping a separate outlier removal step.

---

## Univariate Outlier Methods

_Univariate_ techniques analyze one feature at a time, flagging points that lie far in the tails of its marginal distribution.

### 1. IQR (Boxplot) Method

- **How it works**

  1. Compute the first quartile (Q1, 25th percentile) and third quartile (Q3, 75th percentile).
  2. Interquartile range: IQR = Q3 – Q1.
  3. Define lower bound = Q1 – 1.5 × IQR; upper bound = Q3 + 1.5 × IQR.
  4. Any observation outside \[lower bound, upper bound] is flagged as an outlier.

- **When to use**

  - When you have moderately sized data (n ≈ 50+) and want an interpretable, nonparametric approach.
  - Works well if the distribution is roughly continuous and unimodal.

- **When **not** to use**

  - If distribution is highly skewed or multimodal, standard IQR cutoffs may mark many valid points as outliers.
  - In very small samples (n < 10), quartile estimates can be unstable.

- **Pros**

  - Simple, no strong distributional assumption.
  - Insensitive to extreme values (quartiles are robust).

- **Cons**

  - “1.5×IQR” is heuristic — may under/over‑flag depending on your data.
  - Doesn’t consider feature scale differences—best to apply after log‑transform or scaling for skewed variables.

### 2. Z‑Score Method

- **How it works**

  1. Compute mean (μ) and standard deviation (σ) of the feature.
  2. For each observation x, compute z = (x – μ) / σ.
  3. Flag any x for which |z| > threshold (commonly 2.5 or 3).

- **When to use**

  - If data are approximately Gaussian (bell‑shaped), z‑score efficiently identifies points in extreme tails.
  - Works well for “moderately sized” normally distributed features.

- **When **not** to use**

  - If distribution is heavily skewed or contains extreme outliers: μ and σ themselves may be inflated, masking true outliers.
  - In small samples, estimates of μ and σ are noisy.

- **Pros**

  - Easily interpretable (e.g., |z| > 3 → “within 0.3% quantile tails”).
  - Fast to compute.

- **Cons**

  - **Sensitive to outliers**: a few extreme points inflate σ, making z‑scores smaller for all points, masking outliers.
  - Requires approximate normality—ineffective on long‑tailed or multimodal variables.

### 3. Modified Z‑Score (Median Absolute Deviation)

- **How it works**

  1. Compute the median (m) of the feature.
  2. Compute the median absolute deviation (MAD): MAD = median(|x – m|).
  3. Modified z‑score: M<sub>i</sub> = 0.6745 × (x<sub>i</sub> – m) / MAD.
  4. Flag points with |M<sub>i</sub>| > threshold (common cutoff ≈ 3.5).

- **When to use**

  - When your data contain extreme values that distort μ/σ.
  - More robust than standard z‑score for skewed or heavy‑tailed distributions.

- **When **not** to use**

  - If data are categorical or extremely discrete (e.g., counts of 0,1,2): MAD may be zero or very small.
  - For very small n, median/MAD estimates can be unstable.

- **Pros**

  - **Robust**: median/MAD are unaffected by the largest values.
  - Works better on skewed data than standard z‑score.

- **Cons**

  - MAD = 0 if >50% of values identical or if all values lie at a single point—cannot compute modified z.
  - Requires continuous data (floats).

### 4. Percentile‑Based Cutoffs

- **How it works**

  1. Choose a percentile range—for example, \[1st percentile, 99th percentile].
  2. Flag any values below 1% or above 99% as outliers.

- **When to use**

  - When you want direct control over the fraction of data discarded (e.g., trim top/bottom 0.5%).
  - Helpful for extremely skewed distributions where IQR or z‑score fail.

- **When **not** to use**

  - If you suspect real signal resides in the tails (e.g., rare but valid biological extremes).
  - If distribution has long flat tails—percentile cutoffs may remove too many.

- **Pros**

  - Straightforward to implement; flexible.
  - Can ensure that you remove exactly _x_% of observations.

- **Cons**

  - Arbitrary choice of percentiles—requires domain knowledge.
  - If sample size is small, percentile estimates are noisy.

---

## Bivariate & Pairwise Outlier Analysis

_Bivariate_ methods examine pairs of features and flag points that deviate from the main relationship structure. They help catch outliers that aren’t extreme on any single feature but lie far from the joint distribution.

### 1. Scatterplot Inspect & Visual Thresholds

- **How it works**

  1. Plot feature A vs. feature B in a 2D scatterplot.
  2. Visually identify points that lie far outside the main cluster.
  3. Optionally draw manual cut lines or polygon regions to exclude extremes.

- **When to use**

  - Early exploratory stage to see obvious “clouds” and “islands” of outliers.
  - When looking for bivariate relationships (e.g., gene expression vs. patient age) and curious about extreme deviations.

- **When **not** to use**

  - If you have >10 features to inspect—bivariate manual inspection doesn’t scale.
  - Doesn’t produce a reproducible rule unless you specify thresholds.

- **Pros**

  - Intuitive; good for initial data exploration.
  - Can reveal heteroscedastic patterns or curved relationships.

- **Cons**

  - Subjective; not systematic or reproducible without written thresholds.
  - Time‑consuming for large feature sets.

### 2. Mahalanobis Distance in 2D

- **How it works**

  1. For two continuous features (A, B), compute sample mean vector μ = (μ<sub>A</sub>, μ<sub>B</sub>) and sample covariance matrix Σ.
  2. For each point _x_ = (x<sub>A</sub>, x<sub>B</sub>), Mahalanobis distance =

     $$
       D_M(x) = \sqrt{ (x - μ)^T Σ^{-1} (x - μ) }.
     $$

  3. Compare D<sub>M</sub> to a chi‑square distribution with 2 degrees of freedom: points with D<sub>M</sub>² > χ²<sub>0.975; df=2</sub> (≈ 5.99) flagged as outliers.

- **When to use**

  - When both features are moderately continuous and roughly elliptical in joint distribution.
  - Works well if you expect a roughly bivariate normal cloud.

- **When **not** to use**

  - If A and B are highly skewed or have heavy tails—the covariance underestimates tails, inflating distances.
  - If sample size is very small (< 10), Σ may be singular or unstable.

- **Pros**

  - Accounts for correlation between A and B (unlike simply thresholding each individually).
  - Based on well‑known chi‑square theory for ellipse boundaries.

- **Cons**

  - Sensitive to covariance estimation. Robust covariance estimators (e.g., Minimum Covariance Determinant) may be needed for accuracy in the presence of outliers.
  - Doesn’t scale beyond a handful of features—see multivariate section.

### 3. Regression‑Based Residuals

- **How it works**

  1. Fit a simple model between A and B (e.g., linear regression B \~ A).
  2. Compute residuals ε<sub>i</sub> = B<sub>i</sub> – ŷ<sub>i</sub>.
  3. Flag samples with |ε<sub>i</sub>| above some threshold (e.g., 2–3× σ<sub>ε</sub>) as outliers relative to the trend.

- **When to use**

  - When you expect a clear functional relationship (e.g., gene expression vs. age) and want to spot points that “break” that trend.
  - Suitable if you suspect measurement error in B relative to A.

- **When **not** to use**

  - If no meaningful relationship exists between A and B—a regression B \~ A may be meaningless.
  - If pattern is nonlinear and you fit a linear model, residual method may misclassify valid points as outliers.

- **Pros**

  - Contextual: flags points that deviate from the expected relationship, rather than just extreme marginally.
  - Easy to implement with any regression (linear, polynomial, robust).

- **Cons**

  - Dependent on correct model specification—if relationship is curved but you fit linear, you’ll get many false outliers.
  - Requires enough samples to fit a reliable model.

---

## Multivariate Outlier Methods

When analyzing high‑dimensional data (e.g., >3 features), univariate or bivariate tests fail to capture the joint structure. Multivariate approaches consider all features simultaneously and flag points that lie outside the multivariate distribution’s “core.”

### 1. Mahalanobis Distance (Generalized)

- **How it works**
  Similar to bivariate Mahalanobis, but applied to all selected numeric features.

  $$
    D_M(\mathbf{x}_i) = \sqrt{ (\mathbf{x}_i - \boldsymbol{μ})^T Σ^{-1} (\mathbf{x}_i - \boldsymbol{μ}) },
  $$

  where μ = feature means vector, Σ = covariance matrix.
  Compare D<sub>M</sub>² to χ² distribution with _p_ degrees of freedom (p = number of features), e.g. flag if D<sub>M</sub>² > χ²<sub>0.975; df=p</sub>.

- **When to use**

  - If _p_ (number of numeric features) is modest (< 20–30) and _n_ (samples) is large enough to estimate Σ reliably (_n_ ≫ _p_).
  - Data are roughly multivariate normal after any necessary transformations.

- **When **not** to use**

  - If _p_ is high (near or above _n_), Σ becomes singular or near‑singular—Mahalanobis is unstable.
  - If data exhibit strong nonnormality or nonelliptical clusters: Mahalanobis underestimates tails.

- **Pros**

  - Conceptually straightforward—generalizes univariate outlier detection to multivariate.
  - Rooted in chi‑square theory for determining tail probabilities.

- **Cons**

  - Requires robust covariance estimation when moderate outliers already present (e.g., use Minimum Covariance Determinant).
  - Doesn’t scale well to very high dimensions (curse of dimensionality).

### 2. Principal Component or Robust PCA Methods

- **How it works**

  1. Perform PCA (or Robust PCA) on numeric features.
  2. Identify samples that have extreme scores on the first few principal components (e.g., |PC<sub>1</sub>| or |PC<sub>2</sub>| above a cutoff).
  3. Alternatively, compute Mahalanobis distance in the reduced PC space (keeping top _k_ PCs explaining most variance), then flag extremes.

- **When to use**

  - When _p_ is large (many features) but data lie on a lower‐dimensional manifold.
  - If you suspect that outliers manifest primarily along a combination of features (principal directions).

- **When **not** to use**

  - If important outliers lie in directions captured by later (low‐variance) PCs, you may miss them by only examining top PCs.
  - If data are highly non‑linear and PCA fails to capture structure—consider non‑linear dimensionality reduction (e.g., t‑SNE) for visualization, though not typically used for detection.

- **Pros**

  - Reduces dimensionality first, mitigating covariance singularity issues.
  - Uncovers “directional” extremes—points lying far from the main cluster in PC space.

- **Cons**

  - PCA itself can be influenced by outliers—use Robust PCA (e.g., Minimum Covariance Determinant) if many outliers exist.
  - Interpretation: PC axes are linear combinations—harder to translate back to original features.

### 3. Isolation Forest

- **How it works**

  - Builds an ensemble of random partitioning trees.
  - **Key idea**: Outliers are “easier to isolate”—they require fewer splits to become a singleton.
  - For each sample, average the path length across trees; shorter average path length → higher anomaly (outlier) score.

- **When to use**

  - Large, high‑dimensional datasets where number of features (_p_) may approach or exceed sample size (_n_), but isolation forest scales well.
  - When you have mixed numeric features (works out-of-the-box on continuous data); can be extended to include binary/categorical via one‑hot or ordinal encoding.

- **When **not** to use**

  - If domain knowledge suggests specific linear or parametric structure—anomaly trees may be less interpretable.
  - If data are extremely sparse and high‑dimensional (e.g., one‑hot encoding of thousands of categories), random splits may not “isolate” meaningful anomalies.

- **Pros**

  - No strong distributional assumptions.
  - Scales to high dimensions with linear complexity (O(n × log n)).
  - Requires minimal parameter tuning (mainly number of trees and subsampling size).

- **Cons**

  - Non‑deterministic (random seed); ensure reproducibility by setting seeds.
  - May label points that are simply rare clusters (but not necessarily “erroneous”) as outliers—interpret carefully.

### 4. Local Outlier Factor (LOF)

- **How it works**

  - For each sample _i_, compute the average reachability distance to its _k_ nearest neighbors (density of the local neighborhood).
  - LOF score = (local density of neighbors) / (local density of _i_) (i.e., how much sparser _i_'s neighborhood is compared to its neighbors).
  - A high LOF (> 1) indicates that _i_ is in a relatively sparse region compared to its neighbors—anomaly.

- **When to use**

  - When outliers are local (e.g., a point isolated from its immediate neighbors) rather than global extremes.
  - Works well for datasets with clusters of varying density.

- **When **not** to use**

  - In very high dimensions, distance metrics become less meaningful (curse of dimensionality), degrading LOF performance.
  - For extremely large datasets (n ≫ 10 000), LOF can be slow (computing kNN distances).

- **Pros**

  - Captures local anomalies that global methods (e.g., Mahalanobis) would miss.
  - Unsupervised; no need for labeled anomalies.

- **Cons**

  - Sensitive to choice of _k_ (number of neighbors).
  - Computationally expensive for large n (unless approximate kNN methods employed).

### 5. One‑Class SVM

- **How it works**

  - Train a one‑class Support Vector Machine on the data, learning a decision boundary enclosing most points.
  - Points falling outside this boundary (with negative decision function) flagged as outliers.

- **When to use**

  - When you want a **non‑linear** boundary between “normal” data and anomalies.
  - Works well on moderate‑sized datasets where you suspect anomalies lie outside a complex manifold.

- **When **not** to use**

  - For extremely high‑dimensional data (*p* ≫ *n*), One‑Class SVM can overfit or fail to find a meaningful boundary.
  - If you lack computational resources; kernel SVMs can be slow for large n.

- **Pros**

  - Flexible: kernel choice (RBF, polynomial) can capture non‑linear shapes.
  - Provides an explicit boundary, useful for interpretation.

- **Cons**

  - Requires careful tuning of hyperparameters (e.g., ν, kernel bandwidth).
  - Not easily scalable to large n (because of kernel pairwise computations).

### 6. Clustering‑Based Approaches (e.g., DBSCAN)

- **How it works**

  - Density‑based clustering (DBSCAN) groups data points in regions of high density.
  - Points in low‑density regions (not belonging to any cluster) are labeled “noise” (outliers).

- **When to use**

  - Datasets where clusters have arbitrary shapes and densities; you want to isolate points that do not belong to any dense region.
  - Works in both 2D/3D visualization contexts and higher dimensions (with careful distance metric).

- **When **not** to use**

  - If clusters overlap heavily or have vastly different densities—DBSCAN parameters (eps, min_samples) become tricky to set.
  - In very high dimensions, distance metrics lose meaning—DBSCAN may struggle.

- **Pros**

  - Identifies “noise” as outliers automatically.
  - No need to predefine number of clusters.

- **Cons**

  - Sensitive to hyperparameters (e.g., ε radius, minimum points) which are not always obvious to choose.
  - May classify border points (legitimate but low density) as outliers.

---

## Choosing the Right Method

1. **Data Dimensionality (p) vs. Sample Size (n)**

   - _Univariate/Bivariate_ (IQR, z‑score, scatterplot): suitable when focusing on a small number of features or when you have domain knowledge about which features may outlier.
   - _Multivariate (Mahalanobis, PCA)_: effective when *p* < \<n\_ and data are approximately elliptical.
   - _Tree‑Based (Isolation Forest) & Density‑Based (LOF, DBSCAN)_: scale better to higher _p_, but interpretability can suffer.

2. **Distributional Assumptions**

   - If features are roughly Gaussian (after possible transformations), **z‑score** or **Mahalanobis** are appropriate.
   - For skewed or heavy‑tailed data, prefer **IQR**, **Median/MAD**, or **Isolation Forest** which make fewer parametric assumptions.

3. **Local vs. Global Outliers**

   - Global outliers: points far from the overall distribution—detected by IQR, z‑score, Mahalanobis.
   - Local outliers: points that lie in sparse regions but not necessarily far from the global center—detected by LOF, DBSCAN.

4. **Dimensionality Reduction Preprocessing**

   - If _p_ is large but intrinsic dimensionality is low, perform **PCA/Robust PCA** first, then Mahalanobis or scatterplot in the reduced space.
   - Beware: if outliers drive PCA, use **Robust PCA** or **minimum covariance determinant** estimators.

5. **Computational Budget**

   - **Fast & Simple**: IQR, z‑score, percentile methods—linear time in _n_.
   - **Moderate**: Mahalanobis (requires matrix inversion, O(p³) but p small), PCA (O(n p²) or optimized).
   - **Heavier**: Isolation Forest (O(n log n) per tree), LOF (O(n²) unless using approximate kNN), One‑Class SVM (O(n²)).
   - For very large n (≫ 100k), rely on approximate or sub‑sampled versions of Isolation Forest or LOF.

6. **Interpretability vs. Black‑Box**

   - If you need to document which feature caused an outlier call, **IQR** or **z‑score** on specific features is transparent.
   - Black‑box methods (Isolation Forest, LOF, One‑Class SVM) may give a final score but not easily traceable to one feature.

---

## Treatment Strategies

Once outliers are detected, you must decide **how** to handle them. Common approaches:

### 1. Removal

- **What**: Drop the outlier rows entirely before downstream analysis.
- **When to use**

  - If outliers are clear measurement errors or artifacts.
  - If sample size is large enough that losing a few points won’t harm statistical power.

- **Pros**

  - Simplest approach; guarantees those points won’t skew downstream models.

- **Cons**

  - Risk discarding rare but valid biology.
  - If many outliers exist, dropping them can bias the dataset.

### 2. Imputation (Replace with Estimates)

- **What**: Replace outlier value(s) with a more “typical” value (e.g., median, nearest neighbor).
- **When to use**

  - If you suspect outliers result from data entry errors (typos) or technical glitches that can be “patched.”
  - If you wish to preserve sample count but correct individual feature values.

- **Options**

  - **Univariate**: Replace with median (for that feature).
  - **KNN Impute**: Replace using neighbor feature values (preserves multivariate structure).
  - **Model‑Based**: Fit a regression model to predict that feature from other features; use predicted value.

- **Pros**

  - Maintains full dataset size.
  - May be preferable when downstream models require complete cases.

- **Cons**

  - Introduces “artificial” data points—can bias variance estimates.
  - Harder to justify biologically if the outlier was genuine.

### 3. Winsorizing / Truncation

- **What**: Cap outliers at a specified percentile (e.g., top 1% set to 99th percentile).
- **When to use**

  - When you want to reduce the effect of extreme values but not remove them entirely.
  - Common in financial data or gene expression counts: cap values at reasonable bounds.

- **Pros**

  - Retains sample count; still “softens” the influence of extremes.
  - Simple to implement.

- **Cons**

  - The choice of cutoff is subjective.
  - May still bias distributions if many points are “clipped.”

### 4. Robust Models & Weighting

- **What**: Instead of altering data, use models that down‑weight outlier influence (e.g., Huber regression, Decision trees).
- **When to use**

  - When you prefer to keep all data points but need robustness.
  - Suitable if outlier frequency is low but you’re uncertain whether they’re errors or biology.

- **Pros**

  - No data “editing.”
  - Models learn to reduce the weight of extreme points automatically.

- **Cons**

  - Requires selecting robust algorithm (e.g., Huber loss).
  - May not be enough if outliers cluster in certain dimensions and distort overall covariance.

### 5. Indicator Flags

- **What**: Create binary flag features (1 = outlier, 0 = not) and keep original outlier value (or impute).
- **When to use**

  - When you suspect that “being an outlier” itself is informative (e.g., patients with extreme lab values might have a specific condition).
  - Allows models to learn different patterns for outliers vs. non‑outliers.

- **Pros**

  - Preserves both information: the anomaly and its magnitude.
  - Useful for tree‑based models that can split on flags.

- **Cons**

  - Increases feature space (one flag per detected outlier variable).
  - May confuse models if flags are too frequent.

---

## Common Pitfalls & Best Practices

1. **Mixing Genuine & Technical Outliers**

   - Always investigate outliers to decide if they’re biologically plausible or technical errors.
   - Visualize in context (e.g., scatterplots or PCA colored by batch/condition).
   - If in doubt, flag rather than outright remove; allow downstream analysis to confirm.

2. **Applying Univariate Methods to Multivariate Problems**

   - An observation may be unremarkable on each feature individually but lie far from the joint distribution (e.g., unusual combination of moderate values).
   - Use multivariate methods (Mahalanobis, Isolation Forest, LOF) in high‑dimensional settings.

3. **Overly Aggressive Outlier Removal**

   - Setting IQR cutoff at 3× IQR or z‑threshold at 2 (instead of 3) may remove 5–10% of data, potentially discarding valid extremes.
   - Always report the fraction of data flagged as outliers and inspect a random sample.

4. **Ignoring Feature Scale & Distribution**

   - Always examine raw feature distributions (histogram, log‑transform if skewed) before applying z‑score or Mahalanobis.
   - For skewed data (e.g., gene counts), log-transform first, then apply outlier detection.

5. **Failing to Document Decisions**

   - Keep a record of which features, methods, and thresholds were used, as well as how many observations were flagged/removed.
   - Version control any scripts or notebooks performing outlier detection for reproducibility.

6. **Single Method Mindset**

   - Don’t rely solely on one outlier detection method; combine univariate scans (IQR) with a multivariate check (Isolation Forest) to cross‑validate flagged points.
   - For bivariate relationships, always visualize before and after.

---

## Summary Checklist

Before modeling, run through this quick checklist:

- **Inspect Marginal Distributions**

  - Histograms / boxplots for each numeric feature
  - Are any features heavily skewed? Consider log/MAD first.

- **Apply Univariate Filters**

  - IQR (boxplot) or modified z‑score (MAD) for each numeric column
  - Flag or Winsorize extreme values.

- **Inspect Bivariate Relationships**

  - Scatterplots (e.g., “age vs. some lab value”); check for points far outside main trend.
  - Optionally compute 2D Mahalanobis distances for key feature pairs.

- **Multivariate Screening**

  - If *p* ≈ < 20 and *n* ≫ *p*: compute Mahalanobis distances, flag D<sub>M</sub>² > χ²<sub>0.975; df=p</sub>.
  - For higher dimensions: run Isolation Forest or LOF; examine top‑ranked outliers.

- **Decide Treatment**

  - Technical error → remove or impute.
  - Potential rare biology → flag and investigate (e.g., plot side by side).
  - If using robust models, consider leaving in place but creating an outlier flag.

- **Document All Steps**

  - Record thresholds, methods, number of observations flagged, fraction removed.
  - Save flagged indices to a JSON/CSV manifest (e.g., `reports/outliers/flagged_indices.json`).

- **Re‑Check After Treatment**

  - Recompute PCA or summary stats to ensure that the major data structure remains intact.
  - Confirm that outlier treatment did not inadvertently remove clusters or biological subgroups.

---

By isolating outlier detection and treatment into this dedicated section, you have a clear, structured reference for each stage—from univariate heuristics to advanced multivariate algorithms. Choosing the appropriate method depends on **(a)** your data’s dimensionality and distribution, **(b)** the nature of the suspected outliers, and **(c)** the robustness of your downstream models. Always pair algorithmic flags with domain insight and thorough visualization to ensure you preserve true biological signal while minimizing the undue influence of erroneous extremes.
