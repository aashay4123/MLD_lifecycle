#!/usr/bin/env python3
"""
EDA_combined_v3.py – Enhanced Mathematical EDA (auto vs full)

Usage:
    python EDA_combined_v3.py --input data.csv \
        --outdir eda_v3_reports \
        --mode auto \
        --corr-threshold 0.3 \
        --assoc-threshold 0.2 \
        --normality-alpha 0.05 \
        --bp-alpha 0.05 \
        --max-dendro 30 \
        --sample-size 5000
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, probplot

import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import pingouin as pg

# from pandas_profiling import ProfileReport


def hopkins_statistic(X, m=None, random_state=0):
    """Compute Hopkins statistic for cluster tendency."""
    X = np.asarray(X)
    n, d = X.shape
    m = m or min(100, n // 10)
    random.seed(random_state)
    np.random.seed(random_state)
    # sample points from X
    idx = np.random.choice(np.arange(n), m, replace=False)
    X_m = X[idx]
    # uniform random points in bounding box
    mins, maxs = X.min(axis=0), X.max(axis=0)
    U = np.random.uniform(mins, maxs, size=(m, d))
    # nearest‐neighbor distances
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    du, _ = nbrs.kneighbors(U, return_distance=True)
    dx, _ = nbrs.kneighbors(X_m, return_distance=True)
    # exclude self‐distance (zero)
    dx = dx[:, 1] if dx.shape[1] > 1 else dx[:, 0]
    return float(du.sum() / (du.sum() + dx.sum()))


class EDAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        outdir: str = "reports",
        mode: str = "auto",
        corr_threshold: float = 0.3,
        assoc_threshold: float = 0.2,
        normality_alpha: float = 0.05,
        bp_alpha: float = 0.05,
        max_dendro: int = 30,
        sample_size: int = 5000,
    ):
        self.df = df.copy()
        self.mode = mode.lower()
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.corr_thr = corr_threshold
        self.assoc_thr = assoc_threshold
        self.norm_alpha = normality_alpha
        self.bp_alpha = bp_alpha
        self.max_dendro = max_dendro

        # sample for heavy tests
        if len(self.df) > sample_size:
            self.df_sample = self.df.sample(sample_size, random_state=0)
        else:
            self.df_sample = self.df

        self.report = {}

    @staticmethod
    def _cramers_v(conf):
        chi2 = stats.chi2_contingency(conf)[0]
        n = conf.values.sum()
        r, k = conf.shape
        return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

    @staticmethod
    def _corr_ratio(df, cat, num):
        arr = df[[cat, num]].dropna()
        var_tot = arr[num].var()
        if var_tot == 0 or len(arr) == 0:
            return 0.0
        grps = arr.groupby(cat)[num]
        weighted_var = sum(
            grps.count()[lvl] * grps.var()[lvl] for lvl in grps.groups
        ) / len(arr)
        return max(0.0, 1 - weighted_var / var_tot)

    def univariate(self):
        desc = self.df.describe(include="all").T
        desc["missing"] = self.df.isna().sum()
        desc.to_csv(self.outdir / "univariate_stats.csv")
        self.report["univariate"] = desc

        num_cols = self.df_sample.select_dtypes("number").columns
        norm_res = []
        for col in num_cols:
            vals = self.df_sample[col].dropna()
            if len(vals) < 3:
                continue
            p_sw = shapiro(vals).pvalue
            p_dn = normaltest(vals).pvalue
            p_jb = jarque_bera(vals).pvalue
            norm_res.append(
                {
                    "feature": col,
                    "shapiro_p": p_sw,
                    "dagostino_p": p_dn,
                    "jarque_bera_p": p_jb,
                }
            )
            # QQ-plots: auto if any p < alpha, full always
            if self.mode == "full" or min(p_sw, p_dn, p_jb) < self.norm_alpha:
                plt.figure()
                probplot(vals, plot=plt)
                plt.title(f"QQ-plot {col} (p_min={min(p_sw, p_dn, p_jb):.3f})")
                plt.tight_layout()
                plt.savefig(self.outdir / f"{col}__qq.png")
                plt.close()
        pd.DataFrame(norm_res).to_csv(self.outdir / "normality_tests.csv", index=False)
        self.report["normality"] = norm_res

    def bivariate(self):
        nums = self.df.select_dtypes("number").columns
        cats = self.df.select_dtypes(["object", "category"]).columns
        brep = {"num_num": {}, "num_cat": {}, "cat_cat": {}}

        # numeric–numeric
        for i, x in enumerate(nums):
            for y in nums[i + 1 :]:
                r = self.df[x].corr(self.df[y])
                brep["num_num"][(x, y)] = r
                do_plot = (self.mode == "full") or (abs(r) >= self.corr_thr)
                if do_plot:
                    plt.figure()
                    sns.scatterplot(x=x, y=y, data=self.df, s=10)
                    plt.title(f"{x}↔{y} (pearson r={r:.2f})")
                    plt.tight_layout()
                    plt.savefig(self.outdir / f"{x}__{y}__scatter.png")
                    plt.close()

        # numeric–categorical
        for cat in cats:
            for num in nums:
                eta = self._corr_ratio(self.df, cat, num)
                brep["num_cat"][(cat, num)] = eta
                do_plot = (self.mode == "full") or (eta >= self.assoc_thr)
                if do_plot:
                    plt.figure()
                    sns.boxplot(x=cat, y=num, data=self.df)
                    plt.xticks(rotation=45)
                    plt.title(f"{num} by {cat} (η²={eta:.2f})")
                    plt.tight_layout()
                    plt.savefig(self.outdir / f"{cat}__{num}__box.png")
                    plt.close()

        # categorical–categorical
        for i, a in enumerate(cats):
            for b in cats[i + 1 :]:
                conf = pd.crosstab(self.df[a], self.df[b])
                v = self._cramers_v(conf)
                brep["cat_cat"][(a, b)] = v
                do_plot = (self.mode == "full") or (v >= self.assoc_thr)
                if do_plot:
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(conf, annot=True, fmt="d")
                    plt.title(f"Cramér’s V {a}↔{b} = {v:.2f}")
                    plt.tight_layout()
                    plt.savefig(self.outdir / f"{a}__{b}__heatmap.png")
                    plt.close()

        self.report["bivariate"] = brep

    def multivariate(self):
        nums = self.df.select_dtypes("number").columns.dropna()
        X = self.df[nums].dropna()

        # VIF
        vif = pd.DataFrame(
            {
                "feature": nums,
                "VIF": [
                    variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                ],
            }
        )
        vif.to_csv(self.outdir / "vif.csv", index=False)
        self.report["vif"] = vif

        # PCA scree
        pca = PCA().fit(X)
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("Components")
        plt.ylabel("Cumulative Variance")
        plt.title("PCA Scree")
        plt.tight_layout()
        plt.savefig(self.outdir / "pca_scree.png")
        plt.close()

        # Correlation dendrogram
        do_dendro = (self.mode == "full") or (len(nums) <= self.max_dendro)
        if do_dendro:
            corr = X.corr()
            cg = sns.clustermap(corr, method="average", cmap="vlag")
            cg.fig.suptitle("Correlation Dendrogram")
            plt.tight_layout()
            cg.savefig(self.outdir / "corr_dendrogram.png")
            plt.close()

        # Mardia's multivariate normality
        mH, mP = pg.multivariate_normality(X, alpha=self.norm_alpha)[:2]
        pd.DataFrame([{"H": mH, "p": mP}]).to_csv(
            self.outdir / "mardia.csv", index=False
        )
        self.report["mardia"] = {"H": mH, "p": mP}

        # Hopkins
        H = hopkins_statistic(X)
        pd.DataFrame([{"Hopkins": H}]).to_csv(self.outdir / "hopkins.csv", index=False)
        self.report["hopkins"] = H

        # Breusch–Pagan on first var ~ others
        if len(nums) > 1:
            y = X[nums[0]]
            X_ = add_constant(X[nums[1:]])
            model = OLS(y, X_).fit()
            lm, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)
            pd.DataFrame(
                [{"LM_stat": lm, "LM_p": lm_p, "F_stat": f_stat, "F_p": f_p}]
            ).to_csv(self.outdir / "breuschpagan.csv", index=False)
            self.report["breuschpagan"] = {
                "LM": lm,
                "LM_p": lm_p,
                "F": f_stat,
                "F_p": f_p,
            }

    def advanced(self):
        # Mutual Information
        nums = self.df.select_dtypes("number").columns
        mi = {}
        for col in self.df.columns:
            if col in nums:
                mi[col] = mutual_info_regression(self.df[nums], self.df[col]).mean()
            else:
                codes = self.df[col].astype("category").cat.codes
                mi[col] = mutual_info_classif(self.df[nums], codes).mean()
        pd.Series(mi, name="MI").sort_values(ascending=False).to_csv(
            self.outdir / "mutual_info.csv"
        )
        self.report["mutual_info"] = mi

        # Time-series decomposition + ACF/PACF
        if isinstance(self.df.index, pd.DatetimeIndex):
            from statsmodels.tsa.seasonal import seasonal_decompose

            for col in nums:
                try:
                    res = seasonal_decompose(
                        self.df[col].dropna(), model="additive", period=12
                    )
                    fig = res.plot()
                    fig.suptitle(f"Decompose {col}")
                    plt.tight_layout()
                    fig.savefig(self.outdir / f"{col}__decompose.png")
                    plt.close(fig)
                except:
                    pass
                # ACF/PACF
                series = self.df[col].dropna()
                if len(series) > 10:
                    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
                    plot_acf(series, ax=axes[0], lags=20)
                    plot_pacf(series, ax=axes[1], lags=20)
                    fig.suptitle(f"ACF/PACF {col}")
                    plt.tight_layout()
                    fig.savefig(self.outdir / f"{col}__acf_pacf.png")
                    plt.close(fig)

        # Clustering diagnostics
        Xs = StandardScaler().fit_transform(self.df[nums].dropna())
        ssd, sil = [], []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=0).fit(Xs)
            ssd.append(km.inertia_)
            sil.append(pg.cluster_silhouette_score(Xs, km.labels_))
        # Elbow
        plt.figure()
        plt.plot(range(2, 11), ssd, marker="o")
        plt.title("Elbow Method")
        plt.tight_layout()
        plt.savefig(self.outdir / "elbow.png")
        plt.close()
        # Silhouette
        plt.figure()
        plt.plot(range(2, 11), sil, marker="o")
        plt.title("Silhouette Scores")
        plt.tight_layout()
        plt.savefig(self.outdir / "silhouette.png")
        plt.close()

        # t-SNE
        emb = TSNE(n_components=2, random_state=0).fit_transform(Xs)
        plt.figure()
        plt.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.7)
        plt.title("t-SNE Projection")
        plt.tight_layout()
        plt.savefig(self.outdir / "tsne.png")
        plt.close()

        # HTML profile
        # ProfileReport(self.df, minimal=True).to_file(
        #     self.outdir / "profile.html")

    def _write_manifest(self):
        man = {
            "timestamp": datetime.utcnow().isoformat(),
            "rows": len(self.df),
            "cols": self.df.shape[1],
            "artifacts": [p.name for p in sorted(self.outdir.iterdir())],
        }
        (self.outdir / "manifest.json").write_text(json.dumps(man, indent=2))

    def run(self):
        self.univariate()
        self.bivariate()
        self.multivariate()
        self.advanced()
        self._write_manifest()
        print(f"✅ EDA_v3 complete. Outputs in {self.outdir}/")


"""

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Enhanced Mathematical EDA (auto vs full)")
    p.add_argument("--input", required=True, help="CSV or Parquet file")
    p.add_argument("--outdir", default="eda_v3_output",
                   help="Output directory")
    p.add_argument("--mode", choices=["auto", "full"], default="auto",
                   help="auto: threshold‐based plots; full: all plots/tests")
    p.add_argument("--corr-threshold", type=float, default=0.3,
                   help="Min |pearson r| to plot numeric–numeric (auto)")
    p.add_argument("--assoc-threshold", type=float, default=0.2,
                   help="Min η² or Cramér’s V to plot others (auto)")
    p.add_argument("--normality-alpha", type=float, default=0.05,
                   help="Alpha for normality QQ‐plots (auto)")
    p.add_argument("--bp-alpha", type=float, default=0.05,
                   help="Alpha for Breusch‐Pagan (reported only)")
    p.add_argument("--max-dendro", type=int, default=30,
                   help="Max features for dendrogram (auto)")
    p.add_argument("--sample-size", type=int, default=5000,
                   help="Max rows sampled for heavy tests")
    args = p.parse_args()

    path = Path(args.input)
    if path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type")

    EDAAnalyzerV3(
        df=df,
        outdir=args.outdir,
        mode=args.mode,
        corr_threshold=args.corr_threshold,
        assoc_threshold=args.assoc_threshold,
        normality_alpha=args.normality_alpha,
        bp_alpha=args.bp_alpha,
        max_dendro=args.max_dendro,
        sample_size=args.sample_size,
    ).run()


# ──────────────────────────────────────────────────────────────
# Example Usage ---------------------
# ──────────────────────────────────────────────────────────────


python EDA_combined_v3.py \
  --input data.csv \
  --outdir eda_v3_reports \
  --mode auto \
  --corr-threshold 0.3 \
  --assoc-threshold 0.2 \
  --normality-alpha 0.05 \
  --max-dendro 30 \
  --sample-size 5000

"""
