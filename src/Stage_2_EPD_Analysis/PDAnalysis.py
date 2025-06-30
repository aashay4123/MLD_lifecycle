#!/usr/bin/env python3
from src.utils.perfkit import perfclass
from src.utils.monitor import monitor
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
)
from sklearn.preprocessing import QuantileTransformer
from copulas.multivariate import GaussianMultivariate
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from sklearn.metrics import mutual_info_score
from itertools import combinations
from scipy.stats import entropy as kl_entropy
from statsmodels.distributions.empirical_distribution import ECDF
from configs import global_conf
from pathlib import Path

REPORT_DIR = Path(f"{global_conf.EPDA_REPORT_PATH}/probalistic")


@monitor(name="ProbabilisticAnalysis")
@perfclass
class ProbabilisticAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        mode: str = "auto",
        quantile_output: str = "normal",
        min_dist_count: int = 50,
        entropy_bins: int = 20,
        jobs: int = 1,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame")

        if target and target not in df.columns:
            print(
                f"⚠️ Warning: Target '{target}' not found in dataframe. Ignoring.")
            target = None

        self.df = df.copy().reset_index(drop=True)
        self.target = target
        self.mode = mode.lower()
        self.quantile_output = quantile_output
        self.min_dist_count = min_dist_count
        self.entropy_bins = entropy_bins
        self.jobs = jobs
        self.copula_loglik = None
        self.model_score = None
        self.distributions: dict[str, tuple] = {}
        self.entropy: dict[str, float] = {}
        self.mi_scores: pd.Series | None = None
        self.cond_tables: dict[str, pd.DataFrame] = {}
        self.pit_df: pd.DataFrame | None = None
        self.quantile_df: pd.DataFrame | None = None
        self.group_stats: pd.DataFrame | None = None
        self.copula_model = None
        self.perm_importance_: pd.Series | None = None
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def detect_nonlinearity(x, y):
        # Remove NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 30:
            return False  # too small for reliable test
        x, y = x[mask], y[mask]
        if np.var(x) == 0 or np.var(y) == 0:
            return False  # no variation → can't detect non-linearity

        # Pearson ~ linear, Spearman ~ monotonic (non-linear)
        try:
            pearson_corr, _ = stats.pearsonr(x, y)
            spearman_corr, _ = stats.spearmanr(x, y)
            # Large delta means likely non-linear
            if abs(spearman_corr - pearson_corr) > 0.2:
                return True
        except Exception:
            return True  # fallback to mutual_info if correlation fails

        return False

    def _fit_one_distribution(self, col: str, values: np.ndarray):
        """Helper for parallel distribution fitting on a single column."""

        candidate_distros = [
            "norm",  # Normal (Gaussian) — symmetric bell curve
            "lognorm",  # Log-Normal — positive-skewed data, e.g., income
            "gamma",  # Gamma — positive-only, skewed data rainfall, time
            "beta",  # Beta — bounded between 0 and 1, e.g., proportions, probabilities
            "weibull_min",  # Weibull — versatile, survival/failure analysis
            "expon",  # Exponential — memoryless events, time between Poisson events
            "pareto",  # Pareto — long tail distributions wealth, file sizes
            "t",  # Student's t — ~ normal, but  heavier tails - small samples
            "cauchy",  # Cauchy — very heavy-tailed, no mean/variance, use with caution
            "rayleigh",  # Rayleigh — positive values, magnitude of 2D vectors signal noise
            "triang",  # Triangular — known min/mode/max shape, simple bounded distribution
        ]

        best = None  # (dist_name, params, aic, ks_stat, ks_p)
        for name in candidate_distros:
            dist = getattr(stats, name)
            try:
                if name in {"lognorm", "gamma", "expon", "pareto", "rayleigh"}:
                    if np.any(values <= 0):
                        continue  # requires strictly positive
                elif name in {"beta", "triang"}:
                    if np.any(values < 0) or np.any(values > 1):
                        continue  # must be bounded [0, 1]
                params = dist.fit(values)
                ll = np.sum(dist.logpdf(values, *params))
                k = len(params)
                aic = 2 * k - 2 * ll
                ks_stat, ks_p = stats.kstest(values, name, args=params)
                candidate = (name, params, aic, ks_stat, ks_p)

                if best is None or candidate[2] < best[2]:  # minimize AIC
                    best = candidate
            except Exception:
                continue
            # Add Anderson-Darling test for normality as extra info
        try:
            fitted_cdf = dist.cdf
            ad_stat = self._generic_ad_stat(
                values, lambda x: fitted_cdf(x, *params))
        except Exception:
            ad_stat = None

        if best is None:
            print(f"⚠️ No valid distribution fit found for column '{col}'")

        return col, best, ad_stat

    @monitor(name="fit_all_distributions")
    def fit_all_distributions(self) -> dict[str, tuple]:
        """
        Fit distributions in parallel.
        In 'auto' mode, only columns with >= min_dist_count non-null values are fitted.
        In 'full' mode, all numeric columns (with at least one non-null) are attempted.
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        # filter by count if auto
        if self.mode == "auto":
            numeric_cols = [
                c for c in numeric_cols if self.df[c].count() >= self.min_dist_count
            ]

        # prepare tasks
        tasks = [
            (col, self.df[col].dropna().values)
            for col in numeric_cols
            if len(self.df[col].dropna()) >= 5
        ]

        # parallel execution
        results = Parallel(n_jobs=self.jobs)(
            delayed(self._fit_one_distribution)(self, col, vals) for col, vals in tasks
        )

        # collect best fits
        dist_out = {}
        for col, best, ad_stat in results:
            if best is None:
                continue
            name, params, aic, ks, pval = best
            self.distributions[col] = best
            dist_out[col] = {
                "best_dist": name,
                "params": [float(x) if x is not None else None for x in params],
                "aic": float(aic),
                "ks_stat": float(ks),
                "ks_p": float(pval),
                "anderson_darling": float(ad_stat) if ad_stat is not None else None,
            }

        # save JSON
        path = REPORT_DIR / "best_fit_distributions.json"
        path.write_text(json.dumps(dist_out, indent=2))
        print(f"→ Best-fit distributions written to {path}")
        return self.distributions

    @staticmethod
    def _generic_ad_stat(sample, cdf_fn):
        """Calculate Anderson–Darling statistic for any CDF function."""
        sample = np.sort(sample)
        n = len(sample)
        i = np.arange(1, n + 1)
        cdf_vals = np.clip(cdf_fn(sample), 1e-9, 1 - 1e-9)
        s = -n - np.sum(
            (2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])) / n
        )
        return s

    def shannon_entropy(self) -> dict[str, float]:
        """
        Compute Shannon entropy per column.
        Numeric columns are binned into 'entropy_bins'; categoricals by value frequencies.
        """
        ent = {}
        for col in self.df.columns:
            vals = self.df[col].dropna()
            if vals.empty:
                continue
            if vals.dtype == object or vals.nunique() < self.entropy_bins:
                probs = vals.value_counts(normalize=True).values
                # Avoid log(0) by adding a small epsilon
                ent[col] = float(-np.sum(probs * np.log(probs + 1e-9)))
            else:
                hist, _ = np.histogram(
                    vals.values, bins=self.entropy_bins, density=True
                )
                # Avoid log(0) by adding a small epsilon
                ent[col] = float(stats.entropy(hist + 1e-9))
        self.entropy = ent
        pd.Series(ent, name="shannon_entropy").to_csv(
            REPORT_DIR / "shannon_entropy.csv"
        )
        print(f"→ Shannon entropy saved to {REPORT_DIR/'shannon_entropy.csv'}")
        return ent

    def _compute_correlations_pair(self, c1, c2):
        try:
            x, y = self.df[c1].dropna(), self.df[c2].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 10:  # skip if not enough overlapping data
                return c1, c2, np.nan, np.nan
            tau, _ = stats.kendalltau(x.loc[idx], y.loc[idx])
            spear, _ = stats.spearmanr(x.loc[idx], y.loc[idx])
        except Exception:
            tau, spear = np.nan, np.nan
        return c1, c2, tau, spear

    def cramers_v_matrix(self) -> pd.DataFrame:
        """Pairwise Cramér's V for all object columns (categorical ↔ categorical)."""
        obj_cols = self.df.select_dtypes(include="object").columns
        result = pd.DataFrame(index=obj_cols, columns=obj_cols, dtype=float)

        for col1, col2 in combinations(obj_cols, 2):
            confusion_matrix = pd.crosstab(self.df[col1], self.df[col2])
            if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
                continue  # skip invalid crosstab

            try:
                chi2 = stats.chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                v = np.sqrt(phi2 / min(k - 1, r - 1))
            except ZeroDivisionError:
                v = 0.0

            result.loc[col1, col2] = result.loc[col2, col1] = v

        np.fill_diagonal(result.values, 1.0)
        out_path = REPORT_DIR / "cramers_v_matrix.csv"
        result.to_csv(out_path)
        print(f"→ Cramér’s V matrix saved to {out_path}")
        return result

    def theils_u_matrix(self) -> pd.DataFrame:
        """Theil's U matrix (asymmetrical) for all categorical pairs."""

        def theils_u(x, y):
            s_xy = mutual_info_score(x, y)
            x_enc = pd.factorize(x.dropna())[0]
            hx = stats.entropy(np.bincount(x_enc))
            return s_xy / hx if hx != 0 else 1.0

        obj_cols = self.df.select_dtypes(include="object").columns
        result = pd.DataFrame(index=obj_cols, columns=obj_cols, dtype=float)

        for col1 in obj_cols:
            for col2 in obj_cols:
                if col1 == col2:
                    result.loc[col1, col2] = 1.0
                else:
                    result.loc[col1, col2] = theils_u(
                        self.df[col1], self.df[col2])

        result.to_csv(REPORT_DIR / "theils_u_matrix.csv")
        print(
            f"→ Theil’s U matrix saved to {REPORT_DIR/'theils_u_matrix.csv'}")
        return result

    def rank_correlations(self) -> pd.DataFrame:
        """Kendall's Tau and Spearman correlations for numeric pairs."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        tau_df = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
        spear_df = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)

        results = Parallel(n_jobs=self.jobs)(
            delayed(self._compute_correlations_pair)(c1, c2)
            for c1, c2 in combinations(num_cols, 2)
        )

        for c1, c2, tau, spear in results:
            tau_df.loc[c1, c2] = tau_df.loc[c2, c1] = tau
            spear_df.loc[c1, c2] = spear_df.loc[c2, c1] = spear

        np.fill_diagonal(tau_df.values, 1.0)
        np.fill_diagonal(spear_df.values, 1.0)
        tau_df.to_csv(REPORT_DIR / "kendall_tau.csv")
        spear_df.to_csv(REPORT_DIR / "spearman_corr.csv")
        print("→ Kendall’s Tau and Spearman correlation matrices saved.")
        return tau_df, spear_df

    def mutual_info_scores(self) -> pd.Series | None:
        """
        Compute adaptive dependency scores between features and target.
        Uses f_classif for linear, mutual_info for non-linear relationships.
        """
        if not self.target:
            print("→ No target: skipping mutual_info_scores()")
            return None

        df_nm = self.df.dropna(subset=[self.target])
        y = df_nm[self.target]
        X = pd.get_dummies(df_nm.drop(columns=[self.target]), drop_first=True)
        if X.empty:
            print("→ No features: skipping mutual_info_scores()")
            return None

        method_list = []
        scores = []

        for col in X.columns:
            x_vals = X[col].values
            y_vals = y.values

            # Basic NA filtering
            mask = ~np.isnan(x_vals) & ~pd.isnull(y_vals)
            if np.sum(mask) < 30:
                scores.append(0.0)
                method_list.append("insufficient")
                continue

            x_masked = x_vals[mask].astype(float)
            y_masked = y_vals[mask].astype(float)

            try:
                pearson_corr = stats.pearsonr(x_masked, y_masked)[0]
                spearman_corr = stats.spearmanr(x_masked, y_masked)[0]
            except Exception:
                pearson_corr = 0
                spearman_corr = 0

            # Detect non-linearity
            nonlinear = abs(spearman_corr - pearson_corr) > 0.2

            try:
                if y.nunique() <= 10:  # Classification
                    if nonlinear:
                        score = mutual_info_classif(
                            X[[col]], y, random_state=0)[0]
                        method = "mutual_info"
                    else:
                        score = f_classif(X[[col]], y)[0][0]
                        method = "f_classif"
                else:  # Regression
                    if nonlinear:
                        score = mutual_info_regression(
                            X[[col]], y, random_state=0)[0]
                        method = "mutual_info"
                    else:
                        score = f_classif(X[[col]], y)[0][0]
                        method = "f_classif"
            except Exception:
                score = 0.0
                method = "failed"

            scores.append(score)
            method_list.append(method)

        mi_series = pd.Series(scores, index=X.columns)
        df_out = mi_series.rename("score").to_frame()
        df_out["method"] = method_list
        df_out = df_out.sort_values("score", ascending=False)
        df_out.to_csv(REPORT_DIR / "target_dependency_scores.csv")
        print(
            f"→ Target dependency scores saved to {REPORT_DIR/'target_dependency_scores.csv'}"
        )
        self.mi_scores = mi_series
        return mi_series

    def conditional_probability_tables(self) -> dict[str, pd.DataFrame]:
        """
        Build P(target|category) tables for each object dtype column.
        """
        if not self.target:
            print("→ No target: skipping conditional_probability_tables()")
            return {}

        tables = {}
        for col in self.df.select_dtypes(include="object").columns:
            ct = pd.crosstab(
                self.df[col], self.df[self.target], normalize="index")
            count = self.df[col].value_counts()
            ct["__count"] = count
            tables[col] = ct
            ct.to_csv(REPORT_DIR / f"cpt_{col}.csv")
        print(f"→ CPTs saved to {REPORT_DIR}")
        self.cond_tables = tables
        return tables

    def fit_transform(self) -> pd.DataFrame:
        """
        Probability Integral Transform for each numeric column via ECDF.
        """
        from statsmodels.distributions.empirical_distribution import ECDF

        transformed = {}
        for col in self.df.select_dtypes(include=np.number).columns:
            vals = self.df[col].dropna().values
            if vals.size == 0:
                continue
            ecdf = ECDF(vals)
            transformed[col] = self.df[col].map(
                lambda x: ecdf(x) if pd.notna(x) else np.nan
            )
        pit_df = pd.DataFrame(transformed, index=self.df.index)
        pit_df.to_csv(REPORT_DIR / "pit_transformed.csv", index=False)
        print(f"→ PIT transform saved to {REPORT_DIR/'pit_transformed.csv'}")
        self.pit_df = pit_df
        return pit_df

    def quantile_transform(self) -> pd.DataFrame:
        """
        Quantile transform numeric columns to the specified output_distribution.
        """
        nums = self.df.select_dtypes(include=np.number).columns
        if nums.empty:
            print("→ No numeric cols: skipping quantile_transform()")
            return pd.DataFrame()
        if self.df[nums].shape[0] < 100:
            print("⚠️ Too few rows for quantile transform (<100), skipping.")
            return pd.DataFrame()  # Warning suppressed, but logic flaw noted

        qt = QuantileTransformer(
            output_distribution=self.quantile_output, random_state=0
        )
        arr = qt.fit_transform(self.df[nums])
        qdf = pd.DataFrame(arr, columns=nums, index=self.df.index)
        qdf.to_csv(
            REPORT_DIR / f"quantile_transformed_{self.quantile_output}.csv", index=False
        )
        print(f"→ Quantile transform saved to {REPORT_DIR}")
        self.quantile_df = qdf
        return qdf

    def bayesian_group_comparison(self) -> pd.DataFrame | None:
        """
        For each categorical column vs numeric target, compute group means and 95% CI.
        """
        if not self.target:
            print("→ No target: skipping bayesian_group_comparison()")
            return None

        stats_list = []
        for col in self.df.select_dtypes(include="object").columns:
            grp = self.df.dropna(subset=[self.target]).groupby(col)[
                self.target]
            for name, series in grp:
                arr = series.values
                mu = float(arr.mean())
                ci = np.percentile(arr, [2.5, 97.5])
                stats_list.append(
                    {
                        "feature": col,
                        "group": name,
                        "mean": mu,
                        "ci_lower": float(ci[0]),
                        "ci_upper": float(ci[1]),
                        "std_dev": float(arr.std()),
                        "n": len(arr),
                    }
                )
        gs = pd.DataFrame(stats_list)
        gs.to_csv(REPORT_DIR / "bayesian_group_stats.csv", index=False)
        print(
            f"→ Bayesian group stats saved to {REPORT_DIR/'bayesian_group_stats.csv'}"
        )
        self.group_stats = gs
        return gs

    def copula_modeling(self) -> GaussianMultivariate | None:
        """
        Fit Gaussian copula on all numeric columns (drop NA rows).
        """
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 2:
            print("→ Not enough numeric cols: skipping copula_modeling()")
            return None
        numeric_df = self.df[numeric_cols].dropna()

        model = GaussianMultivariate()
        model.fit(numeric_df)
        path = REPORT_DIR / "copula_params.json"
        path.write_text(json.dumps(model.to_dict(), indent=2))
        print(f"→ Copula params saved to {path}")
        ll = model.log_likelihood(numeric_df)
        self.copula_loglik = ll  # Store for report use
        self.copula_model = model
        return model

    def qq_pp_plots(self, feature: str) -> None:
        """
        Generate side‑by‑side QQ and PP plots for a numeric feature, using best‐fit distribution.
        Saves figure to reports/probabilistic/qqpp_<feature>.png
        """
        if feature not in self.distributions:
            print(
                f"→ Skipping QQ/PP for '{feature}': no fitted distribution found.")
            return

        dist_name, params, _, _, _ = self.distributions[feature]
        values = self.df[feature].dropna().values

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"QQ Plot ({feature})")

        plt.subplot(1, 2, 2)
        sorted_data = np.sort(values)
        ecdf = ECDF(values)
        plt.plot(sorted_data, ecdf(sorted_data), label="Empirical")
        plt.plot(
            sorted_data,
            getattr(stats, dist_name).cdf(sorted_data, *params),
            label=f"{dist_name} CDF",
        )
        plt.legend()
        plt.title(f"PP Plot ({feature})")

        plt.tight_layout()
        out_path = REPORT_DIR / f"qqpp_{feature}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"→ Saved QQ/PP plot for '{feature}' to {out_path}")

    def feature_importance(self) -> pd.Series | None:
        """
        Use a random forest (classification if target has <=10 unique values,
        regression otherwise) to compute permutation importance of every feature.
        Returns a Series {feature: importance}.
        """
        if self.target is None:
            print("→ Skipping feature importance: no target provided.")
            return None
        df_nonmiss = self.df.dropna(subset=[self.target])
        y = df_nonmiss[self.target]
        X = pd.get_dummies(df_nonmiss.drop(
            columns=[self.target]), drop_first=True)
        if X.shape[1] == 0:
            print(
                "→ Skipping feature importance: no features left after one‐hot encoding."
            )
            return None

        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=0)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=0)

        model.fit(X, y)
        self.model_score = model.score(X, y)

        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=0)
        imp_series = pd.Series(perm.importances_mean, index=X.columns).sort_values(
            ascending=False
        )
        out_path = REPORT_DIR / "permutation_importance.csv"
        imp_series.to_csv(out_path, header=["importance"])
        print(f"→ Permutation feature importances written to {out_path}")
        self.perm_importance_ = imp_series
        return imp_series

    def diagnostic_plots(self, feature: str) -> None:
        """
        Plot a histogram with KDE and overlay the fitted PDF (from best‐fit distribution).
        Saves to reports/probabilistic/diagnostic_<feature>.png
        """
        if feature not in self.distributions:
            print(
                f"→ Skipping diagnostic plot for '{feature}': no fitted distribution."
            )
            return

        values = self.df[feature].dropna().values
        dist_name, params, *_ = self.distributions[feature]
        dist = getattr(stats, dist_name)

        plt.figure(figsize=(10, 5))
        sns.histplot(
            values, stat="density", kde=True, color="skyblue", label="Empirical"
        )
        x = np.linspace(values.min(), values.max(), 200)
        plt.plot(
            x, dist.pdf(x, *params), color="darkred", lw=2, label=f"Fitted {dist_name}"
        )
        plt.title(f"Histogram + Fitted PDF for '{feature}'")
        plt.legend()
        plt.tight_layout()
        out_path = REPORT_DIR / f"diagnostic_{feature}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"→ Saved diagnostic plot for '{feature}' to {out_path}")

    def jensen_shannon_divergence(self, p, q):
        """Compute JSD between two distributions (safe wrapper)."""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p + 1e-9, q + 1e-9
        m = 0.5 * (p + q)
        return 0.5 * (kl_entropy(p, m) + kl_entropy(q, m))

    def population_stability_index(self, expected, actual, bins=10):
        """Compute PSI for drift detection between expected and actual."""
        expected_perc, _ = np.histogram(expected, bins=bins, density=True)
        actual_perc, _ = np.histogram(actual, bins=bins, density=True)
        expected_perc = expected_perc + 1e-9
        actual_perc = actual_perc + 1e-9
        return np.sum(
            (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
        )

    def drift_and_divergence_tests(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        results = []

        # For now, use first 10% vs rest as base vs current
        split_idx = int(0.1 * len(self.df))
        base = self.df.iloc[:split_idx]
        current = self.df.iloc[split_idx:]

        for col in num_cols:
            base_vals = base[col].dropna().values
            curr_vals = current[col].dropna().values
            if len(base_vals) < 5 or len(curr_vals) < 5:
                continue
            try:
                jsd = self.jensen_shannon_divergence(
                    np.histogram(base_vals, bins=20, density=True)[0],
                    np.histogram(curr_vals, bins=20, density=True)[0],
                )
                psi = self.population_stability_index(
                    base_vals, curr_vals, bins=20
                )
                kl = kl_entropy(
                    np.histogram(base_vals, bins=20, density=True)[0] + 1e-9,
                    np.histogram(curr_vals, bins=20, density=True)[0] + 1e-9,
                )
                results.append(
                    {"feature": col, "JSD": jsd, "PSI": psi, "KL_divergence": kl}
                )
            except Exception:
                continue

        df_out = pd.DataFrame(results)
        df_out.to_csv(REPORT_DIR / "divergence_tests.csv", index=False)
        print(
            f"→ Drift & divergence tests saved to {REPORT_DIR/'divergence_tests.csv'}"
        )
        return df_out

    def _bootstrap_ci_single(self, col, n_iter, ci_level):
        vals = self.df[col].dropna().values
        if len(vals) < 30:
            return None  # skip unreliable estimates
        samples = np.random.choice(
            vals, size=(n_iter, len(vals)), replace=True)
        means = np.mean(samples, axis=1)
        medians = np.median(samples, axis=1)
        alpha = (100 - ci_level) / 2
        ci_mean = np.percentile(means, [alpha, 100 - alpha])
        ci_median = np.percentile(medians, [alpha, 100 - alpha])
        return {
            "feature": col,
            "mean_lower": ci_mean[0],
            "mean_upper": ci_mean[1],
            "median_lower": ci_median[0],
            "median_upper": ci_median[1],
        }

    def bootstrap_column_ci(self, n_iter=1000, ci_level=95) -> pd.DataFrame:
        """
        For each numeric column, compute bootstrap CI for mean & median.
        More efficient: avoids repeated percentiles inside loop.
        """
        num_cols = self.df.select_dtypes(include=np.number).columns
        results = []
        alpha = (100 - ci_level) / 2

        results = Parallel(n_jobs=self.jobs)(
            delayed(self._bootstrap_ci_single)(col, n_iter, ci_level) for col in num_cols
        )
        results = [r for r in results if r is not None]
        df_out = pd.DataFrame(results)
        df_out.to_csv(REPORT_DIR / "bootstrap_cis.csv", index=False)
        print("→ Bootstrap confidence intervals saved.")
        return df_out

    def generate_final_report(self) -> dict:
        """
        Creates a complete in-memory summary report of all statistical diagnostics.
        Saves both JSON and CSV versions into REPORT_DIR.

        Returns:
            flags: dict of feature → list of tags
        """
        # Ensure skew/kurtosis cached
        if not hasattr(self, "skew_kurt_cache"):
            self.skew_kurt_cache = self.df.select_dtypes(include=np.number).apply(
                lambda x: pd.Series(
                    {
                        "skewness": stats.skew(x.dropna()) if x.nunique() > 1 else 0.0,
                        "kurtosis": (
                            stats.kurtosis(x.dropna(), fisher=True)
                            if x.nunique() > 1
                            else 0.0
                        ),
                    }
                )
            )

        # Ensure drift cached
        if not hasattr(self, "drift_results_cache"):
            if callable(getattr(self, "drift_and_divergence_tests", None)):
                self.drift_results_cache = self.drift_and_divergence_tests()
            else:
                self.drift_results_cache = pd.DataFrame()

        # Ensure entropy calculated
        entropy = self.entropy or self.shannon_entropy()

        flags = {}
        csv_rows = []

        numeric_cols = self.df.select_dtypes(include=np.number).columns
        pit_cols = self.pit_df.columns if self.pit_df is not None else []

        for col in self.df.columns:
            f = []

            # Skew & Kurtosis
            if col in self.skew_kurt_cache.index:
                skew = self.skew_kurt_cache.loc[col, "skewness"]
                kurt = self.skew_kurt_cache.loc[col, "kurtosis"]
                if abs(skew) > 2:
                    f.append("extreme_skew")
                elif abs(skew) > 1:
                    f.append("moderate_skew")
                if abs(kurt) > 5:
                    f.append("extreme_kurtosis")
                elif abs(kurt) > 3:
                    f.append("moderate_kurtosis")

            # Entropy
            if col in entropy:
                if entropy[col] < 0.2:
                    f.append("very_low_entropy")
                elif entropy[col] < 1.0:
                    f.append("low_entropy")

            # Missing best fit
            if col in numeric_cols and col not in self.distributions:
                f.append("no_best_fit_found")

            # PIT coverage
            if col in numeric_cols:
                if col not in pit_cols or self.pit_df[col].isnull().mean() > 0.8:
                    f.append("missing_pit")

            # Drift flags
            if col in self.drift_results_cache["feature"].values:
                drift_row = self.drift_results_cache[
                    self.drift_results_cache["feature"] == col
                ].head(1)
                if not drift_row.empty:
                    psi = drift_row["PSI"].values[0]
                    jsd = drift_row["JSD"].values[0]
                    if psi > 0.3 or jsd > 0.2:
                        f.append("high_drift")
                    elif psi > 0.1 or jsd > 0.1:
                        f.append("moderate_drift")

            # Copula
            if col in numeric_cols and self.copula_model is None:
                f.append("missing_copula")

            flags[col] = f
            csv_rows.append({"feature": col, "flags": ", ".join(f)})

        if self.mi_scores is not None:
            mi_df = pd.read_csv(
                REPORT_DIR / "target_dependency_scores.csv", index_col=0)
            for col, row in mi_df.iterrows():
                if row["method"] == "mutual_info":
                    if col in flags:
                        flags[col].append("nonlinear_dependency")
                        for row_ in csv_rows:
                            if row_["feature"] == col:
                                row_["flags"] += ", nonlinear_dependency"

        # Save reports
        json_path = REPORT_DIR / "final_summary.json"
        csv_path = REPORT_DIR / "final_summary.csv"
        json_path.write_text(json.dumps(flags, indent=2))
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(
            f"✅ Final report written to {json_path.name} and {csv_path.name}")
        return flags

    def run_all(self):

        self.fit_all_distributions()
        self.shannon_entropy()
        self.mutual_info_scores()
        self.conditional_probability_tables()
        self.fit_transform()
        self.quantile_transform()
        self.bayesian_group_comparison()
        self.copula_modeling()
        self.cramers_v_matrix()
        self.theils_u_matrix()
        self.rank_correlations()
        self.bootstrap_column_ci()
        self.drift_results_cache = self.drift_and_divergence_tests()
        self.generate_final_report()
        if self.target:
            self.feature_importance()

        Parallel(n_jobs=self.jobs)(
            delayed(self.diagnostic_plots)(col) for col in self.distributions
        )

        Parallel(n_jobs=self.jobs)(
            delayed(self.qq_pp_plots)(col) for col in self.distributions
        )
