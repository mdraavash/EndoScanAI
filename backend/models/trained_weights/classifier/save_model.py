"""
================================================================================
  save_model.py  —  Save model bundle for classifier_infer.py (web app)
  -----------------------------------------------------------------------
  MUST match 2_validate_and_plot.py EXACTLY:
    - CorrelationLassoSelector WITH f_prefilter_k=40  (best K from sweep)
    - solver="lbfgs"
    - LASSO_CV=3, LASSO_MAX_ITER=20_000
    - class_weight="balanced"

  This is the model whose metrics are reported in the paper:
    AUC=0.6858, Acc=0.6560, Sens=0.6000, Spec=0.6808

  Computes Youden-optimal threshold from 10-fold CV probabilities
  (same method used in extract_all_curves.py plots).

  OUTPUTS:
    backend/models/trained_weights/classifier/model_bundle.pkl

  After running, set in config.py:
    CLASSIFIER_THRESHOLD     = <printed value>
    CLASSIFIER_SELECTOR_PATH = <path to this file>
    CLASSIFIER_MODEL_PATH    = <path to model_bundle.pkl>
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# ── Settings — MUST match 2_validate_and_plot.py exactly ─────────────────────
RADIOMICS_CSV       = "radiomics_features_final.csv"
METADATA_CSV        = "uterus_classification_metadata.csv"
OUT_PATH            = Path("backend/models/trained_weights/classifier/model_bundle.pkl")

CORR_THRESHOLD      = 0.90
LASSO_CV            = 3           # matches 2_validate_and_plot.py
LASSO_MAX_ITER      = 20_000      # matches 2_validate_and_plot.py
FALLBACK_N_FEATURES = 10
RANDOM_STATE        = 42
LR_C                = 1.0
BEST_K              = 40          # best K from the sweep in 2_validate_and_plot.py


# ── CorrelationLassoSelector — exact copy from 2_validate_and_plot.py ─────────
# IMPORTANT: This class MUST live in this file because classifier_infer.py
# loads it as a shim via importlib. Do NOT move or rename this class.
# ─────────────────────────────────────────────────────────────────────────────
class CorrelationLassoSelector(BaseEstimator, TransformerMixin):
    """
    4-stage feature selector — identical to 2_validate_and_plot.py:
      A: Variance filter (drop near-zero variance features)
      B: Pearson correlation filter (drop one from highly correlated pairs)
      C: ANOVA F-score pre-filter (keep top-K into LASSO)
      D: LassoCV (keep non-zero coefficient features)
    """

    def __init__(self,
                 corr_threshold=CORR_THRESHOLD,
                 lasso_cv=LASSO_CV,
                 lasso_max_iter=LASSO_MAX_ITER,
                 f_prefilter_k=BEST_K,
                 fallback_n=FALLBACK_N_FEATURES,
                 random_state=RANDOM_STATE,
                 var_threshold=1e-6):
        self.corr_threshold  = corr_threshold
        self.lasso_cv        = lasso_cv
        self.lasso_max_iter  = lasso_max_iter
        self.f_prefilter_k   = f_prefilter_k
        self.fallback_n      = fallback_n
        self.random_state    = random_state
        self.var_threshold   = var_threshold

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

        # A: variance filter
        variances = np.var(X_arr, axis=0)
        keep_var  = np.where(variances > self.var_threshold)[0]
        X_var     = X_arr[:, keep_var]

        # B: correlation filter
        n_var = X_var.shape[1]
        corr  = np.abs(np.corrcoef(X_var.T))
        upper = np.triu(corr, k=1)
        drop  = set()
        for i in range(n_var):
            for j in range(i + 1, n_var):
                if upper[i, j] > self.corr_threshold and j not in drop:
                    drop.add(j)
        keep_corr_local = [i for i in range(n_var) if i not in drop]
        keep_corr       = [keep_var[i] for i in keep_corr_local]
        X_filt          = X_arr[:, keep_corr]

        # C: F-score pre-filter — top-K into LASSO
        f_scores_pre, _ = f_classif(X_filt, y)
        f_scores_pre    = np.nan_to_num(f_scores_pre, nan=0.0)
        k               = min(self.f_prefilter_k, X_filt.shape[1])
        top_k_local     = np.argsort(f_scores_pre)[::-1][:k]
        keep_topk       = [keep_corr[i] for i in top_k_local]
        X_topk          = X_arr[:, keep_topk]

        # D: LassoCV
        lasso = LassoCV(
            cv=self.lasso_cv,
            random_state=self.random_state,
            max_iter=self.lasso_max_iter,
            eps=1e-4,
            n_alphas=200,
        ).fit(X_topk, y)

        lasso_keep_local    = np.where(lasso.coef_ != 0)[0]
        self.lasso_alpha_   = lasso.alpha_
        self._used_fallback = False

        if len(lasso_keep_local) > 0:
            self.keep_cols_  = [keep_topk[i] for i in lasso_keep_local]
            self.lasso_coef_ = lasso.coef_[lasso_keep_local]
        else:
            # fallback: top-N by F-score
            self._used_fallback = True
            top_n            = min(self.fallback_n, len(keep_topk))
            top_local        = np.argsort(f_scores_pre[top_k_local])[::-1][:top_n]
            self.keep_cols_  = [keep_topk[i] for i in top_local]
            self.lasso_coef_ = np.zeros(len(self.keep_cols_))

        return self

    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return X_arr[:, self.keep_cols_]

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)[self.keep_cols_]
        return np.array(self.keep_cols_)


# ── Pipeline builder — identical to 2_validate_and_plot.py ───────────────────
def build_pipeline(k=BEST_K):
    return Pipeline([
        ("scaler",   StandardScaler()),
        ("selector", CorrelationLassoSelector(f_prefilter_k=k)),
        ("clf",      LogisticRegression(
            penalty="l2", C=LR_C,
            class_weight="balanced",
            solver="lbfgs",           # matches 2_validate_and_plot.py
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def compute_youden_threshold(X, y):
    """
    Compute Youden-optimal threshold from 10-fold CV probabilities.
    This is the same method used in extract_all_curves.py for the
    threshold analysis plot. It gives the operating point that
    maximises Sensitivity + Specificity - 1.
    """
    pipe   = build_pipeline(BEST_K)
    cv     = StratifiedKFold(n_splits=10, shuffle=True,
                             random_state=RANDOM_STATE)
    y_prob = cross_val_predict(pipe, X, y, cv=cv,
                               method="predict_proba")[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    opt_idx   = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[opt_idx])
    return threshold, y_prob


def main():
    print("=" * 65)
    print("  SAVE MODEL BUNDLE — matching 2_validate_and_plot.py")
    print(f"  Pipeline : StandardScaler → CorrelationLassoSelector(K={BEST_K})")
    print(f"             → LogisticRegression(lbfgs, balanced)")
    print(f"  This is the model reported in the paper (AUC=0.6858)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    df   = pd.read_csv(RADIOMICS_CSV)
    meta = pd.read_csv(METADATA_CSV)
    df   = df.merge(meta[["cases.submitter_id", "label"]],
                    left_on="patient_id", right_on="cases.submitter_id")

    feature_cols = [c for c in df.columns
                    if c not in ("patient_id", "cases.submitter_id", "label")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values

    print(f"\n  Patients  : {len(df)}")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  High/Low  : {int(y.sum())} / {int((y==0).sum())}")

    # ── Compute Youden threshold ──────────────────────────────────────────────
    print(f"\n  Computing Youden threshold via 10-fold CV ...")
    threshold, cv_probs = compute_youden_threshold(X, y)

    preds = (cv_probs >= threshold).astype(int)
    acc   = (preds == y).mean()
    sens  = ((preds == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec  = ((preds == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    correct = int((preds == y).sum())

    print(f"  Youden threshold : {threshold:.4f}")
    print(f"  CV Accuracy      : {acc:.4f}")
    print(f"  CV Sensitivity   : {sens:.4f}")
    print(f"  CV Specificity   : {spec:.4f}")
    print(f"  Correct patients : {correct} / {len(y)}")

    # ── Fit full pipeline on all 54 patients ──────────────────────────────────
    print(f"\n  Fitting full pipeline on all {len(y)} patients ...")
    pipe_full = build_pipeline(BEST_K)
    pipe_full.fit(X, y)

    sel           = pipe_full.named_steps["selector"]
    selected_cols = [feature_cols[i] for i in sel.keep_cols_]
    coefs         = sel.lasso_coef_
    fallback_used = getattr(sel, "_used_fallback", False)

    print(f"\n  LASSO selected {len(selected_cols)} features"
          f"{'  [FALLBACK]' if fallback_used else ''}:")
    for name, coef in sorted(zip(selected_cols, coefs),
                              key=lambda x: abs(x[1]), reverse=True):
        direction = "→ High-Grade" if coef > 0 else "→ Low-Grade"
        print(f"    {coef:+.4f}  {name.replace('original_', '')}  {direction}")

    # ── Save bundle ───────────────────────────────────────────────────────────
    # Everything classifier_infer.py needs is stored in one dict.
    # classifier_infer.py loads the pipeline via bundle["pipeline"]
    # and aligns columns via bundle["feature_cols"].
    bundle = {
        # The full fitted sklearn pipeline
        "pipeline"         : pipe_full,

        # ALL 306 feature column names in training order.
        # classifier_infer.py MUST reorder new-patient features to match this.
        "feature_cols"     : feature_cols,

        # The 7 features actually selected by LASSO (subset of feature_cols)
        "selected_cols"    : selected_cols,
        "lasso_coefs"      : coefs,

        # Youden-optimal threshold — use this in config.py
        # Label = "High Grade" if P(High) >= threshold else "Low Grade"
        "threshold"        : threshold,

        # Bookkeeping / provenance
        "n_train_patients" : len(df),
        "train_patient_ids": list(df["patient_id"].values),
        "label_map"        : {0: "Low-Grade", 1: "High-Grade"},
        "best_k"           : BEST_K,
        "solver"           : "lbfgs",
        "lasso_cv"         : LASSO_CV,
        "random_state"     : RANDOM_STATE,
        "reported_auc"     : 0.6858,
        "reported_sens"    : 0.6000,
        "reported_spec"    : 0.6808,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_PATH, compress=3)

    print(f"\n  [SAVED]  {OUT_PATH}")

    # ── Config instructions ───────────────────────────────────────────────────
    print(f"""
  ── Update config.py with these values ────────────────────────
  CLASSIFIER_THRESHOLD     = {threshold:.4f}
  CLASSIFIER_MODEL_PATH    = BACKEND_DIR / "models" / "trained_weights"
                             / "classifier" / "model_bundle.pkl"
  CLASSIFIER_SELECTOR_PATH = BASE_DIR / "save_model.py"
  ──────────────────────────────────────────────────────────────

  ── Inference flow in the web app ─────────────────────────────
  User uploads CT scan (NIfTI)
    → classifier_infer.py extracts radiomics slice-by-slice
    → pools features (mean / max / std per feature)
    → aligns 306 columns to training order
    → pipeline.predict_proba(X)  [scaler → selector → LR]
    → P(High-Grade) compared to threshold {threshold:.4f}
    → returns "High Grade" or "Low Grade" + confidence
  ──────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
