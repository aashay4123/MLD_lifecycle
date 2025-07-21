## 6 — Phase 6 · Model Design & Training

> **Goal**: Pick algorithms, tune hyperparameters, and train your final model(s).
> Implementation is left up to you: you can use `sklearn`, `XGBoost`, `LightGBM`, etc.
> Below is a typical outline:

- **6A** Algorithm Selection (tree‑based vs. linear vs. ensemble).
- **6B** Regularization (L1 / L2 / ElasticNet, early stopping).
- **6C** Cross‑Validation Variants (K‑fold, stratified K‑fold, time‑series CV).
- **6D** Hyperparameter Optimization (GridSearchCV, RandomizedSearchCV, Optuna).
- **6E** Early Stopping & Learning‑Rate Scheduling.
- **6F** Ensembling (Bagging / Stacking / Voting).
- **6G** Data Augmentation & Noise Injection (e.g. mixup for tabular, tiny noise to numeric features).

> **Tip**: Always train (and tune params) on your **train** set only, validate on **val**, and do a final check on **test** only once at the very end. Avoid “peek‑ahead” leakage!

---
