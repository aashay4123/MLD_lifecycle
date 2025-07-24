#!/usr/bin/env python3
"""
Patched EDA_combined_v3.py with parallelization, performance decorators, and figure limits
"""

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.utils.perfkit import PerfMixin, perfclass
from src.utils.monitor import monitor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.api import OLS, add_constant
import pingouin as pg
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import shapiro, normaltest, jarque_bera, probplot
from scipy import stats
from configs import global_conf
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")


def parallel(fn, items, n_jobs=-1, prefer="threads"):
    """Run function fn on each item in items in parallel."""
    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(fn)(item) for item in items)


def hopkins_statistic(X, m=None, random_state=0):
    """Compute Hopkins statistic for cluster tendency."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X)
    n, d = X.shape
    m = m or min(100, n // 10)
    np.random.seed(random_state)

    idx = np.random.choice(np.arange(n), m, replace=False)
    X_m = X[idx]

    mins, maxs = X.min(axis=0), X.max(axis=0)
    U = np.random.uniform(mins, maxs, size=(m, d))

    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    du, _ = nbrs.kneighbors(U, return_distance=True)
    dx, _ = nbrs.kneighbors(X_m, return_distance=True)
    if dx.shape[1] < 1:
        raise ValueError("Hopkins statistic: insufficient neighbors found.")
    dx = dx[:, 1] if dx.shape[1] > 1 else dx[:, 0]

    return float(du.sum() / (du.sum() + dx.sum()))


@perfclass()
class EDAnalyze(PerfMixin):
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
        max_figures: int = 50,
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
        self.max_figures = max_figures

        self.fig_count = 0

        self.df_sample = (
            self.df.sample(sample_size, random_state=0)
            if len(self.df) > sample_size
            else self.df
        )

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

        def norm_test_plot(col):
            vals = self.df_sample[col].dropna()
            if len(vals) < 3:
                return None
            p_sw = shapiro(vals).pvalue
            p_dn = normaltest(vals).pvalue
            p_jb = jarque_bera(vals).pvalue
            if self.fig_count < self.max_figures and (
                self.mode == "full" or min(p_sw, p_dn, p_jb) < self.norm_alpha
            ):
                plt.figure()
                probplot(vals, plot=plt)
                plt.title(f"QQ-plot {col} (p_min={min(p_sw, p_dn, p_jb):.3f})")
                plt.tight_layout()
                plt.savefig(self.outdir / f"{col}__qq.png")
                plt.close()
                self.fig_count += 1
            return {
                "feature": col,
                "shapiro_p": p_sw,
                "dagostino_p": p_dn,
                "jarque_bera_p": p_jb,
            }

        norm_res = parallel(norm_test_plot, list(num_cols))
        norm_res = [r for r in norm_res if r is not None]
        pd.DataFrame(norm_res).to_csv(self.outdir / "normality_tests.csv", index=False)
        self.report["normality"] = norm_res

    def bivariate(self):
        nums = self.df.select_dtypes("number").columns
        cats = self.df.select_dtypes(["object", "category"]).columns
        brep = {"num_num": {}, "num_cat": {}, "cat_cat": {}}

        def process_num_num(pair):
            x, y = pair
            r = self.df[x].corr(self.df[y])
            brep["num_num"][(x, y)] = r
            if self.fig_count < self.max_figures and (
                self.mode == "full" or abs(r) >= self.corr_thr
            ):
                plt.figure()
                sns.scatterplot(x=x, y=y, data=self.df, s=10)
                plt.title(f"{x}↔{y} (r={r:.2f})")
                plt.tight_layout()
                plt.savefig(self.outdir / f"{x}__{y}__scatter.png")
                plt.close()
                self.fig_count += 1

        parallel(
            process_num_num, [(x, y) for i, x in enumerate(nums) for y in nums[i + 1 :]]
        )

        def process_num_cat(pair):
            cat, num = pair
            eta = self._corr_ratio(self.df, cat, num)
            brep["num_cat"][(cat, num)] = eta
            if self.fig_count < self.max_figures and (
                self.mode == "full" or eta >= self.assoc_thr
            ):
                plt.figure()
                sns.boxplot(x=cat, y=num, data=self.df)
                plt.xticks(rotation=45)
                plt.title(f"{num} by {cat} (η²={eta:.2f})")
                plt.tight_layout()
                plt.savefig(self.outdir / f"{cat}__{num}__box.png")
                plt.close()
                self.fig_count += 1

        parallel(process_num_cat, [(cat, num) for cat in cats for num in nums])

        def process_cat_cat(pair):
            a, b = pair
            conf = pd.crosstab(self.df[a], self.df[b])
            v = self._cramers_v(conf)
            brep["cat_cat"][(a, b)] = v
            if self.fig_count < self.max_figures and (
                self.mode == "full" or v >= self.assoc_thr
            ):
                plt.figure(figsize=(6, 5))
                sns.heatmap(conf, annot=True, fmt="d")
                plt.title(f"Cramér’s V {a}↔{b}={v:.2f}")
                plt.tight_layout()
                plt.savefig(self.outdir / f"{a}__{b}__heatmap.png")
                plt.close()
                self.fig_count += 1

        parallel(
            process_cat_cat, [(a, b) for i, a in enumerate(cats) for b in cats[i + 1 :]]
        )
        self.report["bivariate"] = brep

    def multivariate(self):
        nums = self.df.select_dtypes("number").columns.dropna()
        X = self.df[nums].dropna()

        vif = pd.DataFrame(
            {
                "feature": nums,
                "VIF": parallel(
                    lambda i: variance_inflation_factor(X.values, i),
                    list(range(X.shape[1])),
                ),
            }
        )
        vif.to_csv(self.outdir / "vif.csv", index=False)
        self.report["vif"] = vif

        pca = PCA().fit(X)
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("Components")
        plt.ylabel("Cumulative Variance")
        plt.title("PCA Scree")
        plt.tight_layout()
        plt.savefig(self.outdir / "pca_scree.png")
        plt.close()

        if self.mode == "full" or len(nums) <= self.max_dendro:
            corr = X.corr()
            cg = sns.clustermap(corr, method="average", cmap="vlag")
            cg.fig.suptitle("Correlation Dendrogram")
            plt.tight_layout()
            cg.savefig(self.outdir / "corr_dendrogram.png")
            plt.close()

        mH, mP = pg.multivariate_normality(X, alpha=self.norm_alpha)[:2]
        pd.DataFrame([{"H": mH, "p": mP}]).to_csv(
            self.outdir / "mardia.csv", index=False
        )
        self.report["mardia"] = {"H": mH, "p": mP}

        H = hopkins_statistic(X)
        pd.DataFrame([{"Hopkins": H}]).to_csv(self.outdir / "hopkins.csv", index=False)
        self.report["hopkins"] = H

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
        nums = self.df.select_dtypes("number").columns
        mi = {}

        def compute_mi(col):
            if col in nums:
                df_nonan = self.df[nums + [col]].dropna()
                if df_nonan.empty:
                    return (col, np.nan)  # or 0.0 if you prefer
                return (
                    col,
                    mutual_info_regression(df_nonan[nums], df_nonan[col]).mean(),
                )
            else:
                codes = self.df[col].astype("category").cat.codes
                df_nonan = self.df[nums].dropna()
                if df_nonan.empty:
                    return (col, np.nan)
                return (
                    col,
                    mutual_info_classif(df_nonan, codes.loc[df_nonan.index]).mean(),
                )

        mi = dict(parallel(compute_mi, list(self.df.columns)))
        pd.Series(mi, name="MI").sort_values(ascending=False).to_csv(
            self.outdir / "mutual_info.csv"
        )
        self.report["mutual_info"] = mi

        Xs = StandardScaler().fit_transform(self.df[nums].dropna())
        if Xs.shape[0] <= 10000:
            emb = TSNE(n_components=2, random_state=0).fit_transform(Xs)
            plt.figure()
            plt.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.7)
            plt.title("t-SNE Projection")
            plt.tight_layout()
            plt.savefig(self.outdir / "tsne.png")
            plt.close()
        else:
            print("⚠️ Skipping t-SNE: too many rows.")

    def target_analysis(self, target_col: str):
        if target_col is None or target_col not in self.df.columns:
            print("⚠️ Skipping target analysis: target_col not specified or missing.")
            return

        tgt = self.df[target_col]
        plt.figure(figsize=(5, 4))
        if tgt.nunique() <= 10:
            tgt.value_counts(normalize=True).plot.bar()
        else:
            sns.histplot(tgt, kde=True)
        plt.title(f"{target_col} distribution")
        plt.tight_layout()
        plt.savefig(self.outdir / f"{target_col}__distribution.png")
        plt.close()

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        corr_num = []
        for c in num_cols:
            if c == target_col:
                continue
            if tgt.nunique() <= 2:
                r, p = stats.pointbiserialr(self.df[c], tgt)
                corr_num.append({"feature": c, "pointbiserial_r": r, "p": p})
            else:
                r, p = stats.pearsonr(self.df[c], tgt)
                corr_num.append({"feature": c, "pearson_r": r, "p": p})
        pd.DataFrame(corr_num).to_csv(
            self.outdir / "numeric_target_corr.csv", index=False
        )
        self.report["numeric_target_corr"] = corr_num

        cat_v = {}
        for c in cat_cols:
            conf = pd.crosstab(self.df[c], tgt)
            v = self._cramers_v(conf)
            cat_v[c] = v
        pd.Series(cat_v).to_csv(self.outdir / "categorical_target_v.csv")
        self.report["categorical_target_v"] = cat_v

        dates = self.df.select_dtypes(include=["datetime64"]).apply(
            lambda x: x.view(int)
        )
        if not dates.empty and tgt.nunique() <= 2:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(tgt, dates.fillna(0), average="macro")
            self.report["leakage_auc"] = float(auc)
            print(f"🚨 Leakage AUC detected: {auc:.4f}")

    def _write_manifest(self):
        artifacts = [
            str(p.relative_to(self.outdir))
            for p in sorted(self.outdir.glob("*"))
            if p.is_file() and not p.name.endswith(".json")
        ]
        man = {
            "timestamp": datetime.utcnow().isoformat(),
            "rows": len(self.df),
            "cols": self.df.shape[1],
            "num_artifacts": len(artifacts),
            "num_figures": self.fig_count,
            "key_outputs": [
                "univariate_stats.csv",
                "normality_tests.csv",
                "numeric_target_corr.csv",
                "categorical_target_v.csv",
                "vif.csv",
                "mutual_info.csv",
                "mardia.csv",
                "hopkins.csv",
                "breuschpagan.csv",
                "pca_scree.png",
                "corr_dendrogram.png",
                f"{self.df.columns[0]}__distribution.png" if self.df.columns[0] else "",
            ],
            "report_keys": list(self.report.keys()),
        }
        (self.outdir / "manifest.json").write_text(json.dumps(man, indent=2))

    def run_all(self):
        target_col = global_conf.DATASET_TARGET_COLUMN_NAME
        self.univariate()
        self.bivariate()
        self.multivariate()
        self.advanced()
        self._write_manifest()
        if target_col:
            self.target_analysis(target_col)
        print(f"✅ EDA_v3 complete. Outputs in {self.outdir}/")
