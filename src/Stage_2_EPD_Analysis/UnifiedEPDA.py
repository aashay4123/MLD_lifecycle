import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pingouin as pg
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats import diagnostic as sm_diag
from copulas.multivariate import GaussianMultivariate
from joblib import Parallel, delayed

try:
    import dabl
except ImportError:
    dabl = None


def cramers_v(conf: np.ndarray) -> float:
    """
    Cramér's V for categorical–categorical association.
    """
    chi2 = stats.chi2_contingency(conf)[0]
    n = conf.sum().sum()
    r, k = conf.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


def hopkins_stat(df: pd.DataFrame, sample_frac: float = 0.1) -> float:
    """
    Hopkins statistic for cluster tendency (values ≪0.5 indicate clustering).
    """
    X = df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
    n, d = X.shape
    m = max(1, int(sample_frac * n))
    np.random.seed(0)
    idx = np.random.choice(n, m, replace=False)
    S = X.values
    X_m = S[idx]
    nbrs = NearestNeighbors(n_neighbors=2).fit(S)
    ujd = nbrs.kneighbors(
        np.random.uniform(np.min(S, axis=0), np.max(S, axis=0), (m, d)),
        return_distance=True,
    )[0][:, 1]
    wjd = nbrs.kneighbors(X_m, return_distance=True)[0][:, 1]
    return ujd.sum() / (ujd.sum() + wjd.sum())


class ExploratoryDataAnalysis:
    """
    Basic & multivariate exploratory data analysis.
    """

    def __init__(self):
        self.report = {}

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        profile: bool = False,
        pairplots: bool = False,
    ) -> pd.DataFrame:
        self.report.clear()
        # shape, types, missingness, uniques, memory
        self.report["shape"] = df.shape
        self.report["dtypes"] = df.dtypes.apply(lambda x: x.name).to_dict()
        self.report["missing_counts"] = df.isna().sum().to_dict()
        self.report["unique_counts"] = df.nunique().to_dict()
        self.report["memory_MB"] = df.memory_usage(deep=True).sum() / (1024**2)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()

        # univariate numeric
        desc = (
            df[num_cols]
            .describe()
            .T.assign(skew=df[num_cols].skew(), kurtosis=df[num_cols].kurtosis())
        )
        self.report["univariate_numeric"] = desc.to_dict(orient="index")
        # univariate categorical
        self.report["univariate_categorical"] = {
            c: df[c].value_counts().to_dict() for c in cat_cols
        }

        # target distribution
        if target_col and target_col in df.columns:
            tgt = df[target_col]
            plt.figure(figsize=(5, 4))
            if tgt.nunique() <= 10:
                tgt.value_counts(normalize=True).plot.bar()
            else:
                sns.histplot(tgt, kde=True)
            plt.title(f"{target_col} distribution")
            plt.tight_layout()
            plt.savefig("target_distribution.png")
            plt.close()

        # bivariate with target
        if target_col and target_col in df.columns:
            cls = df[target_col]
            corr_num = []
            for c in num_cols:
                if c == target_col:
                    continue
                if cls.nunique() <= 2:
                    r, p = stats.pointbiserialr(df[c], cls)
                    corr_num.append(
                        {"feature": c, "pointbiserial_r": r, "p": p})
                else:
                    r, p = stats.pearsonr(df[c], cls)
                    corr_num.append({"feature": c, "pearson_r": r, "p": p})
            self.report["numeric_target_corr"] = corr_num

            cat_v = {}
            for c in cat_cols:
                conf = pd.crosstab(df[c], cls).values
                cat_v[c] = cramers_v(conf)
            self.report["categorical_target_v"] = cat_v

        # correlation heatmap
        corr_mat = df[num_cols].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_mat, annot=True, fmt=".2f")
        plt.tight_layout()
        plt.savefig("corr_heatmap.png")
        plt.close()
        self.report["corr_matrix"] = corr_mat

        # pairplots
        if pairplots:
            sns.pairplot(df[num_cols + ([target_col] if target_col else [])])
            plt.savefig("pairplots.png")
            plt.close()

        # multivariate stats
        mva = {}
        X = sm.add_constant(df[num_cols].dropna())
        mva["vif"] = {
            col: variance_inflation_factor(X.values, i)
            for i, col in enumerate(X.columns)
            if col != "const"
        }
        _, p_mardia, _ = pg.multivariate_normality(
            df[num_cols].dropna(), alpha=0.05)
        mva["mardia_p"] = float(p_mardia)
        mva["hopkins"] = hopkins_stat(df[num_cols].dropna())
        pca = PCA().fit(df[num_cols].dropna())
        evr = pca.explained_variance_ratio_.cumsum()
        mva["components_90pct"] = int((evr > 0.9).argmax() + 1)
        plt.figure()
        plt.plot(evr)
        plt.axhline(0.9, ls="--")
        plt.tight_layout()
        plt.savefig("pca_scree.png")
        plt.close()
        bp_p = sm_diag.het_breuschpagan(
            df[num_cols].fillna(0).iloc[:, 0], X)[3]
        mva["breusch_pagan_p"] = float(bp_p)

        self.report["multivariate"] = mva

        # leakage sniff
        if target_col and target_col in df.columns:
            dates = df.select_dtypes(include=["datetime64"]).apply(
                lambda x: x.view(int)
            )
            if not dates.empty:
                auc = roc_auc_score(
                    df[target_col], dates.fillna(0), average="macro")
                self.report["leakage_auc"] = float(auc)

        # # HTML profiling
        # if profile and ProfileReport:
        #     ProfileReport(df, explorative=True).to_file("profile.html")
        #     self.report['profile_html'] = "profile.html"

        return df


class AdvancedEDA:
    """
    Advanced EDA visuals & profiling.
    """

    def __init__(self):
        self.report = {}

    def run(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        out_dir: str = "reports/advanced",
    ) -> dict:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        self.report.clear()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()

        # 1. Cramér-V heatmap
        if cat_cols:
            V = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
            for i in cat_cols:
                for j in cat_cols:
                    V.loc[i, j] = cramers_v(pd.crosstab(df[i], df[j]).values)
            plt.figure(figsize=(max(6, len(cat_cols)), max(6, len(cat_cols))))
            sns.heatmap(V, annot=True, fmt=".2f")
            plt.tight_layout()
            plt.savefig(od / "cramers_v_heatmap.png")
            plt.close()
            self.report["cat_assoc_matrix"] = V

        # 2. Mutual Information vs target
        if target_col and target_col in df.columns:
            Xn = df[num_cols].drop(columns=[target_col],
                                   errors="ignore").fillna(0)
            Xm = (
                pd.get_dummies(df[cat_cols], drop_first=True).fillna(0)
                if cat_cols
                else None
            )
            mi_num = (
                mutual_info_regression(
                    Xn, df[target_col].fillna(0)) if num_cols else []
            )
            mi_cat = (
                mutual_info_classif(Xm, df[target_col].fillna(0))
                if Xm is not None
                else []
            )
            mi = pd.Series(
                np.concatenate([mi_num, mi_cat]) if Xm is not None else mi_num,
                index=list(Xn.columns) + (list(Xm.columns)
                                          if Xm is not None else []),
            ).sort_values(ascending=False)
            mi.to_csv(od / "mutual_info.csv")
            plt.figure()
            mi.head(20).plot.barh()
            plt.tight_layout()
            plt.savefig(od / "mutual_info.png")
            plt.close()
            self.report["mutual_info"] = mi

        # 3. Leakage sniff (future-timestamp ratio)
        if target_col and {"last_login", target_col}.issubset(df.columns):
            future_ratio = (
                pd.to_datetime(df["last_login"]) > pd.to_datetime(
                    df[target_col])
            ).mean()
            with open(od / "leakage_check.txt", "w") as f:
                f.write(f"future_ratio: {future_ratio:.4f}\n")
            self.report["leakage_future_ratio"] = float(future_ratio)

        # 4. Time-series decomposition & ACF/PACF
        if "last_login" in df.columns and "amount" in df.columns:
            ts = (
                df.set_index("last_login")["amount"]
                .resample("D")
                .sum()
                .asfreq("D")
                .fillna(method="ffill")
            )
            decomp = sm.tsa.seasonal_decompose(ts, model="additive")
            decomp.plot()
            plt.tight_layout()
            plt.savefig(od / "ts_decompose.png")
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(6, 4))
            sm.graphics.tsa.plot_acf(ts, ax=ax[0])
            sm.graphics.tsa.plot_pacf(ts, ax=ax[1])
            plt.tight_layout()
            plt.savefig(od / "acf_pacf.png")
            plt.close()
            self.report["ts_decomposition"] = decomp

        # 5. KMeans elbow & silhouette
        if num_cols:
            Xs = StandardScaler().fit_transform(df[num_cols].fillna(0))
            ssd, sil = [], []
            for k in range(2, min(10, len(df)) + 1):
                km = KMeans(n_clusters=k, random_state=0).fit(Xs)
                ssd.append(km.inertia_)
                sil.append(pg.cluster_silhouette_score(Xs, km.labels_))
            plt.figure()
            plt.plot(range(2, len(ssd) + 2), ssd, "o-")
            plt.tight_layout()
            plt.savefig(od / "kmeans_elbow.png")
            plt.close()
            plt.figure()
            plt.plot(range(2, len(sil) + 2), sil, "o-")
            plt.tight_layout()
            plt.savefig(od / "kmeans_silhouette.png")
            plt.close()
            self.report["kmeans_ssd"] = ssd
            self.report["kmeans_silhouette"] = sil

        # 6. t-SNE embedding
        if num_cols and target_col and target_col in df.columns:
            emb = TSNE(n_components=2, random_state=0).fit_transform(
                StandardScaler().fit_transform(df[num_cols].fillna(0))
            )
            ed = pd.DataFrame(emb, columns=["tsne1", "tsne2"])
            ed[target_col] = df[target_col].values
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x="tsne1", y="tsne2",
                            hue=target_col, data=ed, s=10)
            plt.tight_layout()
            plt.savefig(od / "tsne_embedding.png")
            plt.close()
            self.report["tsne_embedding"] = ed

        # 7. HTML profiling
        # if ProfileReport:
        #     ProfileReport(df, title="Advanced EDA",
        #                   explorative=True).to_file(od/"profile.html")
        #     self.report['profile_html'] = str(od/"profile.html")

        # 8. dabl quick plot
        if dabl:
            try:
                dabl.plot(df, target_col=target_col)
                plt.tight_layout()
                plt.savefig(od / "dabl_plot.png")
                plt.close()
                self.report["dabl_plot"] = str(od / "dabl_plot.png")
            except Exception:
                pass

        return self.report


class ProbabilisticAnalysis:
    """
    Probability-based analysis.
    """

    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df.copy().reset_index(drop=True)
        self.target = target if target in df.columns else None
        self.imputed: pd.DataFrame = None
        self.distributions = {}
        self.entropy = {}
        self.mi_scores = None

    def impute_missing(self, method: str = "simple") -> pd.DataFrame:
        df = self.df.copy()
        if method == "knn":
            imp = KNNImputer()
            df[df.select_dtypes(include=[np.number]).columns] = imp.fit_transform(
                df.select_dtypes(include=[np.number])
            )
        else:
            df = df.fillna(df.median(numeric_only=True)).fillna("missing")
        self.imputed = df
        return df

    def fit_all_distributions(self) -> dict:
        """Fit candidate dists on each numeric col in parallel."""
        cols = self.numeric_df.columns.tolist()
        if not cols:
            return {}

        def _fit_one(col):
            data = self.numeric_df[col].values
            best = (None, np.inf, None)
            for dist in (
                stats.norm,
                stats.expon,
                stats.gamma,
                stats.beta,
                stats.lognorm,
            ):
                try:
                    params = dist.fit(data)
                    aic = 2 * (len(params) + 1) - 2 * \
                        dist.logpdf(data, *params).sum()
                    if aic < best[1]:
                        best = (dist.name, aic, params)
                except Exception:
                    continue
            return col, best

        results = Parallel(n_jobs=-1)(delayed(_fit_one)(c) for c in cols)
        self.distributions = dict(results)
        return self.distributions

    def shannon_entropy(self) -> dict:
        ent = {}
        for col in self.df.columns:
            p, _ = np.histogram(
                self.df[col].dropna(), bins="auto", density=True)
            p = p[p > 0]
            ent[col] = -np.sum(p * np.log2(p))
        self.entropy = ent
        return ent

    def mutual_info_scores(self) -> pd.Series | None:
        """Compute MI per feature in parallel."""
        if not self.target or self.imputed is None:
            return None

        X = self.imputed.drop(columns=[self.target], errors="ignore").fillna(0)
        cols = X.columns.tolist()
        if not cols:
            return pd.Series(dtype=float)

        y = self.imputed[self.target]
        y_enc = (
            LabelEncoder().fit_transform(y)
            if y.dtype.kind not in "ifu" or y.nunique() <= 20
            else y.values
        )

        def _mi(col):
            arr = X[col].values.reshape(-1, 1)
            if y.dtype.kind in "ifu" and y.nunique() > 20:
                score = mutual_info_regression(
                    arr, y_enc, discrete_features=False)[0]
            else:
                score = mutual_info_classif(
                    arr, y_enc, discrete_features=True)[0]
            return col, float(score)

        results = Parallel(n_jobs=-1)(delayed(_mi)(c) for c in cols)
        self.mi_scores = pd.Series({col: sc for col, sc in results})
        return self.mi_scores

    def conditional_probability_tables(self) -> dict[str, pd.DataFrame]:
        if not self.target:
            return {}

        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if not cat_cols:
            return {}

        def _cpt(col):
            tbl = pd.crosstab(
                self.df[col], self.df[self.target], normalize="index")
            return col, tbl

        results = Parallel(n_jobs=-1)(delayed(_cpt)(c) for c in cat_cols)
        return dict(results)

    def pit_transform(self) -> pd.DataFrame:
        df_pit = pd.DataFrame()
        for col, (name, _, params) in self.distributions.items():
            dist = getattr(stats, name)
            df_pit[col] = dist.cdf(self.imputed[col], *params)
        return df_pit

    def quantile_transform(self, output_distribution: str = "normal") -> pd.DataFrame:
        qt = QuantileTransformer(output_distribution=output_distribution)
        arr = qt.fit_transform(self.imputed.select_dtypes(include=[np.number]))
        return pd.DataFrame(
            arr, columns=self.imputed.select_dtypes(
                include=[np.number]).columns
        )

    def bayesian_group_comparison(self, col: str) -> dict:
        from scipy.stats import ttest_ind

        groups = self.df.groupby(self.target)[col].apply(list)
        a, b = groups.iloc[0], groups.iloc[1] if len(groups) > 1 else ([], [])
        stat, p = ttest_ind(a, b, equal_var=False) if b else (np.nan, np.nan)
        return {"t_stat": float(stat), "p_value": float(p)}

    def predictive_intervals(self, col: str, alpha: float = 0.05) -> tuple:
        mu, sigma = self.df[col].mean(), self.df[col].std()
        lo = stats.norm.ppf(alpha / 2, loc=mu, scale=sigma)
        hi = stats.norm.ppf(1 - alpha / 2, loc=mu, scale=sigma)
        return lo, hi

    def copula_modeling(self) -> GaussianMultivariate:
        model = GaussianMultivariate()
        model.fit(self.imputed.select_dtypes(include=[np.number]))
        return model

    def qq_pp_plots(self, feature: str) -> None:
        sm.qqplot(self.df[feature].dropna(), line="s")
        plt.savefig(f"qq_{feature}.png")
        plt.close()
        plt.figure()
        stats.probplot(self.df[feature].dropna(), plot=plt)
        plt.savefig(f"pp_{feature}.png")
        plt.close()

    def feature_importance(self) -> dict:
        if not self.target:
            return {}
        X = self.imputed.drop(columns=[self.target], errors="ignore")
        y = self.imputed[self.target]
        if y.dtype.kind in "ifu" and y.nunique() > 20:
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
        model.fit(X.fillna(0), y.fillna(0))
        return dict(zip(X.columns, model.feature_importances_))

    def diagnostic_plots(self, feature: str) -> None:
        plt.figure()
        sns.histplot(self.df[feature].dropna(), kde=True)
        plt.savefig(f"hist_{feature}.png")
        plt.close()
        plt.figure()
        stats.probplot(self.df[feature].dropna(), plot=plt)
        plt.savefig(f"diag_{feature}.png")
        plt.close()


class UnifiedEPDA:
    """
    Unified Exploratory Probabilistic Data Analysis:
    Combines basic EDA, advanced EDA, and probabilistic analysis into one report.
    """

    def __init__(
        self, df: pd.DataFrame, target: str = None, out_dir: str = "reports/unified"
    ):
        self.df = df.copy().reset_index(drop=True)
        self.target = target if target in df.columns else None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sample_frac = 1.0
        sns.set(style="whitegrid")
        self.report = {}

        self.basic = ExploratoryDataAnalysis()
        self.adv = AdvancedEDA()
        self.prob = ProbabilisticAnalysis(self.df, self.target)

    def run(
        self,
        profile: bool = False,
        pairplots: bool = False,
        impute_method: str = "simple",
    ) -> dict:
        # Basic EDA
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if not num_cols:
            self.report["warning"] = "no numeric columns – skipping numeric analyses"
            return self.df  # or continue
        if not cat_cols:
            self.report["warning_cat"] = (
                "no categorical columns – skipping categorical analyses"
            )
            return self.df  # or continue
        if self.sample_frac < 1.0:
            self.df = self.df.sample(frac=self.sample_frac, random_state=0).reset_index(
                drop=True
            )

        self.basic.analyze(self.df, self.target,
                           profile=profile, pairplots=pairplots)
        self.report["basic"] = self.basic.report

        # Advanced EDA
        adv_out = self.adv.run(
            self.df, self.target, out_dir=str(self.out_dir / "advanced")
        )
        self.report["advanced"] = adv_out

        # Probabilistic Analysis
        self.prob.impute_missing(method=impute_method)
        dists = self.prob.fit_all_distributions()
        ent = self.prob.shannon_entropy()
        mi = self.prob.mutual_info_scores()
        cpt = self.prob.conditional_probability_tables()
        pit = self.prob.pit_transform()
        qt = self.prob.quantile_transform()
        bgc = {
            col: self.prob.bayesian_group_comparison(col)
            for col in self.df.select_dtypes(include=[np.number]).columns
        }
        preds = {
            col: self.prob.predictive_intervals(col)
            for col in self.df.select_dtypes(include=[np.number]).columns
        }
        cop = self.prob.copula_modeling()
        fi = self.prob.feature_importance()

        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.prob.diagnostic_plots(col)

        self.report["probabilistic"] = {
            "distributions": dists,
            "entropy": ent,
            "mutual_info": mi,
            "conditional_prob_tables": cpt,
            "pit": pit,
            "quantile_transform": qt,
            "bayesian_group_comparison": bgc,
            "predictive_intervals": preds,
            "copula_model": cop,
            "feature_importance": fi,
        }

        # save manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat(),
            "rows": int(len(self.df)),
            "features": int(self.df.shape[1]),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return self.report
