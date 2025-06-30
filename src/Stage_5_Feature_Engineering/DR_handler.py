#!/usr/bin/env python3
"""
full_auto_dr_pipeline.py — Final Industry-Grade Dimensionality Handler
Author: ChatGPT • 2025-06-27

Features:
- Identifies correlated features (|r| > 0.8)
- Runs AutoDR separately on correlated & all numeric features
- Evaluates explained variance, PC orthogonality, predictive power (AUC)
- Visualizes scree, PC correlation, class separation
- Picks best DR strategy automatically
- Saves reduced data + pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from .Dimensionality_Reduction import AutoDR
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv("your_preprocessed_data.csv")
    target_col = "cancer_label"  # change as needed

    # ================================
    # STEP 1: Identify numeric features
    # ================================
    num_cols = df.select_dtypes(include="number").columns.drop(target_col).tolist()

    # ================================
    # STEP 2: Identify highly correlated features (|r| > 0.8)
    # ================================
    cor = df[num_cols].corr().abs()
    upper = np.triu(np.ones(cor.shape), k=1).astype(bool)
    cor_pairs = cor.where(upper)
    correlated_cols = set()
    for col in cor.columns:
        high_corr = cor_pairs[col][cor_pairs[col] > 0.8].index.tolist()
        if high_corr:
            correlated_cols.add(col)
            correlated_cols.update(high_corr)
    correlated_cols = list(correlated_cols)
    print(f"Highly correlated columns: {correlated_cols or 'None found'}")

    # ================================
    # STEP 3A: AutoDR on correlated features
    # ================================
    if correlated_cols:
        print("\n==== Running AutoDR on CORRELATED columns ====")
        df_corr = df[correlated_cols + [target_col]].copy()
        dr_corr = AutoDR(target=target_col, verbose=True)
        df_corr_red = dr_corr.fit_transform(df_corr)
        expl_corr = dr_corr.report[dr_corr.chosen_technique]["info"].get(
            "explained_variance", 0
        )
        print(f"[Correlated] Explained Variance: {expl_corr:.2%}")

        X_corr, y_corr = (
            df_corr_red.drop(columns=[target_col]).values,
            df_corr_red[target_col].values,
        )
        auc_corr = cross_val_score(
            LogisticRegression(solver="liblinear", random_state=0),
            X_corr,
            y_corr,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
            scoring="roc_auc",
            n_jobs=-1,
        ).mean()
        print(f"[Correlated] CV AUC: {auc_corr:.3f}")
    else:
        dr_corr, df_corr_red, expl_corr, auc_corr = None, None, 0, 0

    # ================================
    # STEP 3B: AutoDR on all numeric features
    # ================================
    print("\n==== Running AutoDR on ALL numeric columns ====")
    df_all = df[num_cols + [target_col]].copy()
    dr_all = AutoDR(target=target_col, verbose=True)
    df_all_red = dr_all.fit_transform(df_all)
    expl_all = dr_all.report[dr_all.chosen_technique]["info"].get(
        "explained_variance", 0
    )
    print(f"[All Features] Explained Variance: {expl_all:.2%}")

    X_all, y_all = (
        df_all_red.drop(columns=[target_col]).values,
        df_all_red[target_col].values,
    )
    auc_all = cross_val_score(
        LogisticRegression(solver="liblinear", random_state=0),
        X_all,
        y_all,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring="roc_auc",
        n_jobs=-1,
    ).mean()
    print(f"[All Features] CV AUC: {auc_all:.3f}")

    # ================================
    # STEP 4: Select best DR
    # ================================
    if auc_corr >= auc_all:
        print("\n✅ Best strategy: CORRELATED columns DR")
        best_dr, best_df = dr_corr, df_corr_red
    else:
        print("\n✅ Best strategy: ALL columns DR")
        best_dr, best_df = dr_all, df_all_red

    print(f"Selected DR Technique: {best_dr.chosen_technique}")
    print(f"Detailed report: {best_dr.report}")

    # ================================
    # STEP 5: Visualize & Validate PCA
    # ================================
    pc_cols = [c for c in best_df.columns if c.startswith(best_dr.chosen_technique)]
    scaler, model = best_dr.models.get(best_dr.chosen_technique, (None, None))

    if model and hasattr(model, "explained_variance_ratio_"):
        # Scree plot
        cum_var = model.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title(f"Scree plot: {best_dr.chosen_technique}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("final_scree_plot.png")
        plt.close()
        print("✔️ Saved scree plot as final_scree_plot.png")

    if len(pc_cols) >= 2:
        # Class separation plot
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x=best_df[pc_cols[0]],
            y=best_df[pc_cols[1]],
            hue=best_df[target_col],
            palette="coolwarm",
            alpha=0.7,
        )
        plt.xlabel(pc_cols[0])
        plt.ylabel(pc_cols[1])
        plt.title(f"Class separation: {pc_cols[0]} vs {pc_cols[1]}")
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.savefig("final_class_separation.png")
        plt.close()
        print("✔️ Saved class separation plot as final_class_separation.png")

        # PC correlation heatmap
        pc_df = pd.DataFrame(best_df[pc_cols])
        pc_corr = pc_df.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(pc_corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Correlation matrix of principal components")
        plt.tight_layout()
        plt.savefig("final_pc_correlation.png")
        plt.close()
        mean_corr = (
            pc_corr.where(~np.eye(pc_corr.shape[0], dtype=bool)).abs().mean().mean()
        )
        print(f"✔️ Saved PC correlation plot as final_pc_correlation.png")
        print(f"✔️ Mean absolute off-diagonal PC correlation: {mean_corr:.3f}")

    # Final predictive performance on reduced data
    aucs = cross_val_score(
        LogisticRegression(solver="liblinear", random_state=0),
        best_df[pc_cols],
        best_df[target_col],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"✔️ Final CV AUC on best DR: {aucs.mean():.3f} ± {aucs.std():.3f}")

    # ================================
    # STEP 6: Save Outputs
    # ================================
    best_df.to_csv("train_reduced.csv", index=False)
    with open("dr_pipeline.pkl", "wb") as f:
        pickle.dump(best_dr, f)
    print(
        "\n🎉 Saved reduced dataset (train_reduced.csv) and DR pipeline (dr_pipeline.pkl)"
    )


if __name__ == "__main__":
    main()
