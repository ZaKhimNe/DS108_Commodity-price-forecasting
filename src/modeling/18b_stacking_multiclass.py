"""
18b_stacking_multiclass.py — Tầng 3b: Stacking Multiclass (down/flat/up)
==========================================================================
Meta-learner (Logistic Regression multinomial) kết hợp 4 base models.
Meta-features: 4 models × 3 prob cols = 12 features.
Target: target_multiclass (hard int 0/1/2) từ val/test set.

Prerequisite:
  models/14b_lgbm_multiclass/val_predictions_{tag}.csv  (y_prob_lgb_{down,flat,up})
  models/15b_rf_multiclass/val_predictions_{tag}.csv    (y_prob_rf_{down,flat,up})
  models/16b_lstm_multiclass/val_predictions_{tag}.csv  (y_prob_lstm_{down,flat,up})
  models/17b_tcn_multiclass/val_predictions_{tag}.csv   (y_prob_tcn_{down,flat,up})
  + test_predictions_{tag}.csv cho mỗi model
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

LGBM_DIR   = os.path.join(PROJECT_ROOT, "models", "14b_lgbm_multiclass")
RF_DIR     = os.path.join(PROJECT_ROOT, "models", "15b_rf_multiclass")
LSTM_DIR   = os.path.join(PROJECT_ROOT, "models", "16b_lstm_multiclass")
TCN_DIR    = os.path.join(PROJECT_ROOT, "models", "17b_tcn_multiclass")
TENSOR_DIR = os.path.join(PROJECT_ROOT, "data",   "tensors_multiclass_soft")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "18b_stacking_multiclass")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

CLASS_NAMES = ["down", "flat", "up"]

# Prob cols per model: 3 per model × 4 models = 12 meta-features
PROB_COL_GROUPS = {
    "lgbm": [f"y_prob_lgb_{c}"  for c in CLASS_NAMES],
    "rf":   [f"y_prob_rf_{c}"   for c in CLASS_NAMES],
    "lstm": [f"y_prob_lstm_{c}" for c in CLASS_NAMES],
    "tcn":  [f"y_prob_tcn_{c}"  for c in CLASS_NAMES],
}

MODEL_DIRS = {
    "lgbm": LGBM_DIR,
    "rf":   RF_DIR,
    "lstm": LSTM_DIR,
    "tcn":  TCN_DIR,
}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_val_preds(tag, model_key):
    """Load val_predictions CSV for one model. Returns DataFrame or None."""
    path = os.path.join(MODEL_DIRS[model_key], f"val_predictions_{tag}.csv")
    if not os.path.exists(path):
        print(f"   [MISS] {model_key} val_predictions: {os.path.basename(path)}")
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df


def load_test_preds_with_date(tag, model_key):
    """
    Load test_predictions CSV and attach Date if not present.
    LGBM/RF already have Date. LSTM/TCN need it from test_metadata.parquet.
    """
    pred_path = os.path.join(MODEL_DIRS[model_key], f"test_predictions_{tag}.csv")
    if not os.path.exists(pred_path):
        print(f"   [MISS] {model_key} test_predictions: {os.path.basename(pred_path)}")
        return None

    df = pd.read_csv(pred_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        return df

    # No Date — retrieve from results JSON + test_metadata.parquet
    result_path = os.path.join(MODEL_DIRS[model_key], f"results_{tag}.json")
    if not os.path.exists(result_path):
        print(f"   [MISS] {model_key} results JSON")
        return None

    with open(result_path) as f:
        res = json.load(f)
    best_w = res.get("best_window")
    if best_w is None:
        print(f"   [MISS] best_window in {model_key} results")
        return None

    meta_path = os.path.join(TENSOR_DIR, f"win_{best_w}", tag, "test_metadata.parquet")
    if not os.path.exists(meta_path):
        print(f"   [MISS] test_metadata.parquet (win={best_w}) for {tag}")
        return None

    meta_df   = pd.read_parquet(meta_path)
    if "Date" in meta_df.columns:
        raw_dates = pd.to_datetime(meta_df["Date"].values, utc=True)
    elif isinstance(meta_df.index, pd.DatetimeIndex):
        idx = meta_df.index
        raw_dates = idx.tz_localize("UTC") if idx.tz is None else idx
    else:
        print(f"   [WARN] No Date in metadata for {tag}/{model_key}")
        return None

    n_pred, n_meta = len(df), len(raw_dates)
    if n_pred != n_meta:
        n = min(n_pred, n_meta)
        df        = df.iloc[-n:].reset_index(drop=True)
        raw_dates = raw_dates[-n:]

    df.insert(0, "Date", raw_dates)
    return df


# ─── Merge ────────────────────────────────────────────────────────────────────

def merge_on_date(dfs_dict):
    """
    Inner-join DataFrames in dfs_dict by Date.
    dfs_dict: {model_key: DataFrame with Date + prob_cols}
    Returns merged DataFrame or None if fewer than 2 models.
    """
    available = {k: v for k, v in dfs_dict.items() if v is not None}
    if len(available) < 2:
        print(f"   [SKIP] Chỉ có {len(available)} model — cần ít nhất 2.")
        return None

    merged = None
    for model_key, df in available.items():
        cols = ["Date"] + PROB_COL_GROUPS[model_key]
        cols_present = [c for c in cols if c in df.columns]
        sub = df[cols_present].copy()
        if merged is None:
            if "y_true" in df.columns:
                sub["y_true"] = df["y_true"].values
            merged = sub
        else:
            merged = merged.merge(sub, on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def build_meta_features(merged):
    """Collect all available prob cols as meta-feature matrix."""
    all_prob_cols = []
    for model_key in MODEL_DIRS:
        for c in PROB_COL_GROUPS[model_key]:
            if c in merged.columns:
                all_prob_cols.append(c)
    return all_prob_cols


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(name, y_true, probs, y_pred):
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")) \
              if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    dist = {int(c): int((y_true == c).sum()) for c in sorted(np.unique(y_true))}
    print(
        f"   [{name:<20}]  n={len(y_true):>4}  dist={dist}"
        f"  OvR-AUC={auc:.4f}  MacroF1={f1:.4f}  Acc={acc:.4f}"
    )
    return dict(
        model=name, n=int(len(y_true)), dist=dist,
        auc_roc=float(auc) if not np.isnan(auc) else None,
        f1=float(f1), accuracy=float(acc),
    )


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(tag):
    print(f"\n{'═'*64}")
    print(f"  {tag.upper()} — Stacking Multiclass")
    print(f"{'═'*64}")

    # ── OOF meta-train: val predictions ──
    val_dfs = {k: load_val_preds(tag, k) for k in MODEL_DIRS}
    meta_train = merge_on_date(val_dfs)

    # ── Meta-test: test predictions ──
    test_dfs = {k: load_test_preds_with_date(tag, k) for k in MODEL_DIRS}
    meta_test = merge_on_date(test_dfs)

    if meta_train is None or meta_test is None:
        return None

    prob_cols = build_meta_features(meta_train)
    prob_cols = [c for c in prob_cols if c in meta_test.columns]

    if "y_true" not in meta_train.columns:
        print(f"   [SKIP] meta-train thiếu y_true — không thể fit meta-learner.")
        return None
    if "y_true" not in meta_test.columns:
        print(f"   [SKIP] meta-test thiếu y_true.")
        return None

    if len(meta_train) < 5 or len(prob_cols) < 3:
        print(f"   [WARN] meta-train quá nhỏ hoặc ít features. Bỏ qua.")
        return None

    X_tr = meta_train[prob_cols].values.astype(np.float32)
    y_tr = meta_train["y_true"].values.astype(np.int64)
    X_te = meta_test[prob_cols].values.astype(np.float32)
    y_te = meta_test["y_true"].values.astype(np.int64)

    print(f"   meta-train={len(meta_train)} (OOF val)  meta-test={len(meta_test)}")
    print(f"   Meta-features ({len(prob_cols)}): {prob_cols}")
    print(f"   Train dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"   Test  dist: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    if len(np.unique(y_tr)) < 2:
        print(f"\n   [SKIP] meta-train chỉ có 1 class — không thể fit LR.")
        return None

    # ── Train meta-learner ──
    meta_lr = LogisticRegression(
        solver="lbfgs", C=1.0,
        max_iter=500, random_state=42, class_weight="balanced",
    )
    meta_lr.fit(X_tr, y_tr)

    probs_tr = meta_lr.predict_proba(X_tr)  # (N, 3)
    probs_te = meta_lr.predict_proba(X_te)
    y_pred_tr = np.argmax(probs_tr, axis=1)
    y_pred_te = np.argmax(probs_te, axis=1)

    # ── Individual model results (on meta-test, argmax threshold) ──
    print(f"\n   Individual base learners on meta-test:")
    base_results = {}
    best_base_auc = 0.0
    for model_key in MODEL_DIRS:
        cols = [c for c in PROB_COL_GROUPS[model_key] if c in meta_test.columns]
        if len(cols) != 3:
            continue
        p = meta_test[cols].values.astype(np.float32)
        y_pred_base = np.argmax(p, axis=1)
        r = evaluate(model_key, y_te, p, y_pred_base)
        base_results[model_key] = r
        best_base_auc = max(best_base_auc, r["auc_roc"] or 0.0)

    # ── Meta-learner results ──
    print(f"\n   Meta-learner (multinomial LR stacking) on meta-test:")
    meta_res = evaluate("stacking_lr", y_te, probs_te, y_pred_te)

    delta = (meta_res["auc_roc"] or 0.0) - best_base_auc
    sign  = "+" if delta >= 0 else ""
    print(f"\n   ΔAUC vs best individual: {sign}{delta:.4f}")

    # ── Save ──
    model_path  = os.path.join(OUTPUT_DIR, f"meta_lr_{tag}.joblib")
    pred_path   = os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv")
    result_path = os.path.join(OUTPUT_DIR, f"results_{tag}.json")

    joblib.dump(meta_lr, model_path)

    out_df = meta_test[["Date"] + [c for c in prob_cols if c in meta_test.columns]
                       + ["y_true"]].copy()
    out_df["y_pred_stack"] = y_pred_te
    for i, cls in enumerate(CLASS_NAMES):
        out_df[f"y_prob_stack_{cls}"] = probs_te[:, i]
    out_df.to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "tag":              tag,
            "n_meta_train":     int(len(meta_train)),
            "n_meta_test":      int(len(meta_test)),
            "prob_cols":        prob_cols,
            "delta_auc_vs_best_individual": float(delta),
            "base_results":     base_results,
            "stacking":         meta_res,
        }, f, indent=2)

    print(f"\n   Saved: {os.path.basename(model_path)}")
    return {**meta_res, "delta_auc_vs_best": float(delta)}


if __name__ == "__main__":
    all_results = {}
    for tag in TAGS:
        try:
            res = run_pipeline(tag)
            if res:
                all_results[tag] = res
        except Exception as e:
            print(f"\n[ERROR] {tag}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        print(f"\n\n{'═'*64}")
        print("  SUMMARY — Stacking Multiclass")
        print(f"{'═'*64}")
        print(f"  {'Tag':<36} {'OvR-AUC':>9} {'ΔvsBase':>9} {'F1-mac':>8}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            name  = tag.replace("integrated_", "")
            auc   = res.get("auc_roc")   or 0.0
            delta = res.get("delta_auc_vs_best", 0.0)
            f1    = res.get("f1", 0.0)
            sign  = "+" if delta >= 0 else ""
            print(f"  {name:<36}  {auc:>9.4f}  {sign}{delta:>8.4f}  {f1:>8.4f}")
        print(f"\n  Predictions saved to: {OUTPUT_DIR}")
    else:
        print("\n  Đảm bảo đã chạy 14b–17b multiclass scripts trước.")
