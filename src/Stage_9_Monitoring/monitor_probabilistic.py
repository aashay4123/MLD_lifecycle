import logging
from pathlib import Path

# # ──── 1A  Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
from src.data_analysis.Probabilistic.probabilistic_analysis import ProbabilisticAnalysis

# # ──── 1B  Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("probabilistic_monitor")
# # ──── 2A  Data Loading ─────────────────────────────────────────
DATA_DIR = Path("data/interim")
RAW = DATA_DIR / "clean.parquet"
# Load the cleaned dataset
df = pd.read_parquet(RAW)

# # ──── 2B  Probabilistic Analysis Setup ────────────────────────
if df.empty:
    raise ValueError("DataFrame is empty. Cannot perform analysis.")

pa = ProbabilisticAnalysis(df)

# --- Distribution Drift Detection ---
dist_results = pa.fit_all_distributions()
for feature, result in dist_results.items():
    if result["ks_stat"] > 0.2:
        raise ValueError(f"Drift detected in {feature}: KS={result['ks_stat']:.3f}")

# --- Entropy Collapse Alert (e.g. categorical bottleneck) ---
entropy_scores = pa.shannon_entropy()
for col, score in entropy_scores.items():
    if score < 0.5:
        log.warning(f"{col} shows entropy collapse: H={score:.3f}")

# --- Mutual Information Drop (low predictivity of target) ---
mi_scores = pa.mutual_info_scores(target="target_col")
for feat, score in mi_scores.items():
    if score < 0.01:
        log.warning(f"Low mutual information: {feat} → target = {score:.4f}")

# --- Copula Joint Distribution Check (advanced use case) ---
try:
    model = pa.copula_modeling()
    log.info("Copula joint model fitted successfully.")
except Exception as e:
    log.warning(f"Copula modeling failed: {e}")

# --- Predictive Interval Collapse (low variance = overconfidence) ---
for feature in df.select_dtypes(include=np.number).columns:
    low, high = pa.predictive_intervals(feature)
    if high - low < 0.1:
        log.warning(
            f"Predictive interval too narrow for {feature}: [{low:.3f}, {high:.3f}]"
        )

# --- Feature Importance Anomaly (target drift?) ---
imp = pa.feature_importance("target_col")
for f, v in imp.items():
    if v < 0.005:
        log.warning(f"Feature {f} has near-zero importance ({v:.4f})")
