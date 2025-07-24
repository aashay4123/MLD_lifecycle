from joblib import Parallel, delayed
from scipy.stats import shapiro, kurtosis, skew, entropy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from scipy import stats
from scipy.stats import shapiro, kurtosis, skew, entropy, norm
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils import resample

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from src.utils.monitor import monitor
from src.utils.perfkit import PerfMixin, perfclass
from configs import global_conf
import warnings

warnings.filterwarnings("ignore")

tqdm.pandas()
sns.set(style="whitegrid")

MAX_PLOTS = 50

N_JOBS = os.cpu_count() or 4


def safe_apply(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


@perfclass()
class UnifiedEPDA(PerfMixin):
    def __init__(
        self,
        df: pd.DataFrame,
        target: str = None,
        mode: str = "auto",
        out_dir: str = global_conf.EPDA_REPORT_PATH,
    ):
        super().__init__()
        self.df = df.copy()
        self.target = target
        self.mode = "auto" if mode not in ["auto", "full"] else mode
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.start_time = datetime.now()

        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if target and target in df.columns:
            self.y = df[target]
        else:
            self.y = None

        self.plot_counter = 0
        self.report = defaultdict(dict)
        self._limit_features_for_runtime()

    def _limit_features_for_runtime(self):
        if self.mode == "auto":
            self.numeric_cols = (
                self.df[self.numeric_cols]
                .std()
                .sort_values(ascending=False)
                .head(100)
                .index.tolist()
            )
            self.categorical_cols = self.categorical_cols[:20]

    def _save_plot(self, fig, dir, name):
        if self.plot_counter >= MAX_PLOTS:
            plt.close(fig)
            return

        out = Path(f"{self.out_dir}/{dir}")
        out.mkdir(exist_ok=True, parents=True)
        path = out / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        self.plot_counter += 1
        plt.close(fig)

    def _run_descriptive_stats(self):
        desc = self.df.describe(include="all").transpose()
        desc["missing_ratio"] = self.df.isnull().mean()
        desc["skew"] = self.df[self.numeric_cols].skew()
        desc["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        desc["dtype"] = self.df.dtypes
        desc.to_csv(self.out_dir / "basic_descriptive_stats.csv")

    def _run_missing_visuals(self):
        miss_ratio = self.df.isnull().mean().sort_values(ascending=False)
        miss_ratio = miss_ratio[miss_ratio > 0]
        if not miss_ratio.empty and self.plot_counter < MAX_PLOTS:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=miss_ratio.index, y=miss_ratio.values, ax=ax)
            ax.set_title("Missing Value Ratio per Column")
            ax.tick_params(axis="x", rotation=90)
            self._save_plot(fig, "other", "missing_ratio_bar")

    def _run_correlation_heatmap(self):
        """Correlation heatmap for numeric variables."""
        if len(self.numeric_cols) < 2 or self.plot_counter >= MAX_PLOTS:
            return
        corr = self.df[self.numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        self._save_plot(fig, "other", "correlation_heatmap")

    def _run_normality_tests(self):
        """Run Shapiro and Jarque-Bera tests for normality."""
        results = []
        for col in self.numeric_cols:
            values = self.df[col].dropna()
            if len(values) < 20:
                continue
            try:
                shapiro_p = shapiro(values)[1]
                jb_p = stats.jarque_bera(values)[1]
                results.append((col, shapiro_p, jb_p))
            except Exception:
                continue
        norm_df = pd.DataFrame(results, columns=["column", "shapiro_p", "jb_p"])
        norm_df.to_csv(self.out_dir / "normality_tests.csv", index=False)

    def _run_vif_analysis(self):
        X = self.df[self.numeric_cols].dropna()
        vif_data = []
        if X.shape[1] >= 2:
            for i in range(X.shape[1]):
                vif = variance_inflation_factor(X.values, i)
                vif_data.append((X.columns[i], vif))
            vif_df = pd.DataFrame(vif_data, columns=["feature", "VIF"])
            vif_df.to_csv(self.out_dir / "vif_report.csv", index=False)

    def _run_pca_analysis(self):
        try:
            X = self.df[self.numeric_cols].dropna()
            pca = PCA()
            pca.fit(X)
            var_ratio = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(x=range(1, len(var_ratio) + 1), y=var_ratio, marker="o", ax=ax)
            ax.set_title("PCA Scree Plot")
            ax.set_xlabel("Component")
            ax.set_ylabel("Explained Variance Ratio")
            self._save_plot(fig, "other", "pca_scree")
        except Exception:
            pass

    def _hopkins_statistic(self, X):
        """Hopkins statistic implementation for clustering tendency."""
        try:
            from sklearn.neighbors import NearestNeighbors
            from numpy.random import uniform

            X = X.dropna().sample(min(100, len(X)), random_state=1)
            n, d = X.shape
            m = int(0.1 * n)
            nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

            rand_pts = uniform(X.min().values, X.max().values, (m, d))
            u_dist, _ = nbrs.kneighbors(rand_pts, 2, return_distance=True)
            x_dist, _ = nbrs.kneighbors(X.sample(m).values, 2, return_distance=True)

            H = u_dist[:, 0].sum() / (u_dist[:, 0].sum() + x_dist[:, 0].sum())
            return H
        except Exception:
            return np.nan

    def _run_clustering_analysis(self):
        """KMeans + Elbow + Silhouette analysis."""
        try:
            X = self.df[self.numeric_cols].dropna()
            inertias, silhouettes = [], []
            for k in range(2, 7):
                model = KMeans(n_clusters=k, random_state=1).fit(X)
                inertias.append(model.inertia_)
                from sklearn.metrics import silhouette_score

                silhouettes.append(silhouette_score(X, model.labels_))

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.lineplot(x=range(2, 7), y=inertias, ax=ax[0], marker="o")
            ax[0].set_title("KMeans Inertia")
            sns.lineplot(x=range(2, 7), y=silhouettes, ax=ax[1], marker="o")
            ax[1].set_title("Silhouette Scores")
            self._save_plot(fig, "other", "kmeans_elbow_silhouette")

            hopkins = self._hopkins_statistic(X)
            with open(self.out_dir / "hopkins_score.txt", "w") as f:
                f.write(f"Hopkins Statistic: {hopkins:.4f}\n")
        except Exception:
            pass

    def _run_tsne_projection(self):
        """Run t-SNE projection on numeric columns."""
        try:
            X = self.df[self.numeric_cols].dropna()
            if X.shape[1] < 2 or len(X) > 1000:
                X = X.sample(n=1000, random_state=42)
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
            proj = tsne.fit_transform(X)
            proj_df = pd.DataFrame(proj, columns=["TSNE1", "TSNE2"])
            fig, ax = plt.subplots()
            sns.scatterplot(data=proj_df, x="TSNE1", y="TSNE2", ax=ax)
            ax.set_title("t-SNE Projection")
            self._save_plot(fig, "other", "tsne_projection")
        except Exception:
            pass

    def _plot_distribution_and_box(self, col):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(self.df[col].dropna(), kde=True, ax=axs[0])
        axs[0].set_title(f"Histogram - {col}")
        sns.boxplot(x=self.df[col], ax=axs[1])
        axs[1].set_title(f"Boxplot - {col}")
        self._save_plot(fig, "hist", f"hist_box_{col}")

    def _plot_qq(self, col):
        fig = plt.figure()
        stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
        plt.title(f"QQ Plot - {col}")
        self._save_plot(fig, "qq", f"qq_{col}")

    def _plot_acf_pacf(self, col):
        values = self.df[col].dropna().values
        if len(values) < 30:
            return
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].stem(acf(values, nlags=20))
        axs[0].set_title(f"ACF - {col}")
        axs[1].stem(pacf(values, nlags=20))
        axs[1].set_title(f"PACF - {col}")
        self._save_plot(fig, "acf_pacf", f"acf_pacf_{col}")

    def _run_parallel_distributions(self):
        Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._plot_distribution_and_box, col)
            for col in self.numeric_cols[:MAX_PLOTS]
        )

    def _run_parallel_qq(self):
        Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._plot_qq, col)
            for col in self.numeric_cols[:MAX_PLOTS]
        )

    def _run_parallel_acf_pacf(self):
        Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._plot_acf_pacf, col)
            for col in self.numeric_cols[:MAX_PLOTS]
        )

    def _compute_entropy_for_col(self, col):
        from scipy.stats import gaussian_kde

        values = self.df[col].dropna().values
        kde = gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 100)
        ps = kde(xs)
        ps /= ps.sum()
        return col, entropy(ps)

    def _run_parallel_entropy(self):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._compute_entropy_for_col, col)
            for col in self.numeric_cols
        )
        entropy_dict = {k: v for k, v in results if k is not None}
        pd.Series(entropy_dict).sort_values(ascending=False).to_csv(
            self.out_dir / "entropy_scores.csv"
        )

    def _compute_pit_for_col(self, col):
        qt = QuantileTransformer(output_distribution="uniform")
        values = self.df[[col]].dropna()
        if values.shape[0] < 10:
            return None
        pit = qt.fit_transform(values)
        fig, ax = plt.subplots()
        sns.histplot(pit.flatten(), kde=True, ax=ax, bins=20)
        ax.set_title(f"PIT Histogram - {col}")
        self._save_plot(fig, "pit", f"pit_{col}")

    def _run_parallel_pit(self):
        Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._compute_pit_for_col, col)
            for col in self.numeric_cols[:MAX_PLOTS]
        )

    def _compute_mutual_info_parallel(self):
        if self.y is None:
            return
        x_data = self.df[self.numeric_cols].fillna(0)
        if self.y.nunique() < 20:
            mi = mutual_info_classif(x_data, self.y)
        else:
            mi = mutual_info_regression(x_data, self.y)
        pd.Series(mi, index=self.numeric_cols).sort_values(ascending=False).to_csv(
            self.out_dir / "mutual_info.csv"
        )

    def _compare_groups_bayesian(self, col):
        try:
            groups = self.df[[col, self.target]].dropna().groupby(self.target)
            if groups.ngroups != 2:
                return None
            g1, g2 = [groups.get_group(k)[col].values for k in groups.groups]
            diff = np.mean(g1) - np.mean(g2)
            p_val = stats.ttest_ind(g1, g2, equal_var=False).pvalue
            return col, diff, p_val
        except Exception:
            return None

    def _run_parallel_bayesian(self):
        if self.y is None:
            return
        results = Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._compare_groups_bayesian, col)
            for col in self.numeric_cols
        )
        df = pd.DataFrame(
            [r for r in results if r], columns=["feature", "mean_diff", "p_value"]
        )
        df.to_csv(self.out_dir / "bayesian_group_comparison.csv", index=False)

    def _fit_best_distribution(self, col):
        from scipy.stats import norm, expon, gamma, beta, lognorm

        dists = [norm, expon, gamma, beta, lognorm]
        values = self.df[col].dropna()
        best_fit = None
        best_aic = np.inf
        for dist in dists:
            try:
                params = dist.fit(values)
                loglik = np.sum(dist.logpdf(values, *params))
                k = len(params)
                aic = 2 * k - 2 * loglik
                if aic < best_aic:
                    best_fit = dist.name
                    best_aic = aic
            except Exception:
                continue
        return col, best_fit, best_aic

    def _run_parallel_distribution_fit(self):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._fit_best_distribution, col)
            for col in self.numeric_cols
        )
        df = pd.DataFrame(
            [r for r in results if r], columns=["feature", "best_fit", "AIC"]
        )
        df.to_csv(self.out_dir / "distribution_fit_quality.csv", index=False)

    def _detect_drift(self, col):
        from scipy.spatial.distance import jensenshannon

        try:
            groups = self.df.groupby(self.y)[col]
            dists = []
            for _, g in groups:
                values = g.dropna().values
                if len(values) < 10:
                    continue
                kde = stats.gaussian_kde(values)
                xs = np.linspace(values.min(), values.max(), 100)
                ps = kde(xs)
                ps /= ps.sum()
                dists.append(ps)
            if len(dists) == 2:
                jsd = jensenshannon(dists[0], dists[1])
                return col, jsd
        except Exception:
            return None

    def _run_parallel_drift_detection(self):
        if self.y is None:
            return
        results = Parallel(n_jobs=N_JOBS)(
            delayed(safe_apply)(self._detect_drift, col) for col in self.numeric_cols
        )
        drift_scores = {k: v for k, v in results if k}
        pd.Series(drift_scores).sort_values(ascending=False).to_csv(
            self.out_dir / "drift_scores.csv"
        )

    def _run_copula_modeling(self):
        try:
            from copulas.multivariate import GaussianMultivariate

            data = self.df[self.numeric_cols].dropna()
            if data.shape[1] < 2:
                return
            model = GaussianMultivariate()
            model.fit(data)
            synthetic = model.sample(data.shape[0])
            synthetic.to_csv(self.out_dir / "copula_sample.csv", index=False)
        except Exception:
            return

    def _run_cpt_tables(self):
        if not self.categorical_cols or self.y is None:
            return
        for col in self.categorical_cols:
            try:
                crosstab = pd.crosstab(self.df[col], self.y, normalize="index")
                crosstab.to_csv(self.out_dir / f"cpt_{col}.csv")
            except Exception:
                continue

    def _run_feature_importance(self):
        if self.y is None:
            return
        X = self.df[self.numeric_cols].fillna(0)
        try:
            if self.y.nunique() < 20:
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()
            model.fit(X, self.y)
            importances = pd.Series(model.feature_importances_, index=self.numeric_cols)
            importances.sort_values(ascending=False).to_csv(
                self.out_dir / "feature_importance.csv"
            )
        except Exception:
            return

    @monitor(name="UnifiedEPDA")
    def run(self):
        print(
            f"[UnifiedEPDA] Running in {self.mode.upper()} mode — {len(self.df)} rows, {self.df.shape[1]} columns"
        )
        self.start_time = datetime.now()

        # ========== BASIC EDA ==========
        self._run_descriptive_stats()
        self._run_missing_visuals()
        self._run_correlation_heatmap()
        self._run_parallel_distributions()
        self._run_parallel_qq()
        self._run_parallel_acf_pacf()
        # ========== ADVANCED EDA ==========
        self._run_normality_tests()
        self._run_vif_analysis()
        self._run_pca_analysis()
        self._run_clustering_analysis()
        self._run_tsne_projection()

        # ========== PROBABILISTIC ANALYSIS ==========
        self._run_parallel_entropy()
        self._compute_mutual_info_parallel()
        self._run_cpt_tables()
        self._run_parallel_pit()
        self._run_parallel_bayesian()
        self._run_copula_modeling()
        self._run_parallel_drift_detection()
        self._run_parallel_distribution_fit()
        self._run_feature_importance()

        # ========== REPORT ==========
        duration = datetime.now() - self.start_time
        manifest = {
            "n_rows": len(self.df),
            "n_columns": self.df.shape[1],
            "target": self.target,
            "mode": self.mode,
            "runtime_seconds": duration.total_seconds(),
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "plots_generated": self.plot_counter,
        }
        pd.Series(manifest).to_json(self.out_dir / "epda_manifest.json", indent=4)

        print(
            f"[UnifiedEPDA] Completed in {round(duration.total_seconds(), 2)} seconds"
        )
        return manifest
