"""
18c_stacking_regression.py — Tầng 3c: Stacking Regression
==========================================================
Ridge meta-learner kết hợp 4 base models.
Meta-features: 4 models × 1 pred col = 4 features.
Target: y_true (continuous return_future) từ val/test set.

Prerequisite:
  models/14c_lgbm_regression/val_predictions_{tag}.csv   (y_pred_lgb_reg)
  models/15c_rf_regression/val_predictions_{tag}.csv     (y_pred_rf_reg)
  models/16c_lstm_regression/val_predictions_{tag}.csv   (y_pred_lstm_reg)
  models/17c_tcn_regression/val_predictions_{tag}.csv    (y_pred_tcn_reg)
  + test_predictions_{tag}.csv cho mỗi model
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LassoCV, Ridge
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

LGBM_DIR   = os.path.join(PROJECT_ROOT, "models", "14c_lgbm_regression")
RF_DIR     = os.path.join(PROJECT_ROOT, "models", "15c_rf_regression")
LSTM_DIR   = os.path.join(PROJECT_ROOT, "models", "16c_lstm_regression")
TCN_DIR    = os.path.join(PROJECT_ROOT, "models", "17c_tcn_regression")
TENSOR_DIR = os.path.join(PROJECT_ROOT, "data",   "tensors_reg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "18c_stacking_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

# 1 pred col per model
PRED_COL_MAP = {
    "lgbm": "y_pred_lgb_reg",
    "rf":   "y_pred_rf_reg",
    "lstm": "y_pred_lstm_reg",
    "tcn":  "y_pred_tcn_reg",
}

MODEL_DIRS = {
    "lgbm": LGBM_DIR,
    "rf":   RF_DIR,
    "lstm": LSTM_DIR,
    "tcn":  TCN_DIR,
}

SIGNAL_THR = 0.05


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_val_preds(tag, model_key):
    path = os.path.join(MODEL_DIRS[model_key], f"val_predictions_{tag}.csv")
    if not os.path.exists(path):
        print(f"   [MISS] {model_key} val_predictions: {os.path.basename(path)}")
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df


def load_test_preds_with_date(tag, model_key):
    """Load test_predictions and attach Date if not present (LSTM/TCN)."""
    pred_path = os.path.join(MODEL_DIRS[model_key], f"test_predictions_{tag}.csv")
    if not os.path.exists(pred_path):
        print(f"   [MISS] {model_key} test_predictions: {os.path.basename(pred_path)}")
        return None

    df = pd.read_csv(pred_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        return df

    # No Date — retrieve from results JSON + test_metadata.parquet (LSTM/TCN)
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
    available = {k: v for k, v in dfs_dict.items() if v is not None}
    if len(available) < 2:
        print(f"   [SKIP] Chỉ có {len(available)} model — cần ít nhất 2.")
        return None

    merged = None
    for model_key, df in available.items():
        pred_col = PRED_COL_MAP[model_key]
        cols     = ["Date"] + ([pred_col] if pred_col in df.columns else [])
        if "y_true" in df.columns and merged is None:
            cols.append("y_true")
        sub = df[[c for c in cols if c in df.columns]].copy()
        if merged is None:
            merged = sub
        else:
            merge_cols = [c for c in sub.columns if c != "y_true"]
            merged = merged.merge(sub[merge_cols], on="Date", how="inner")

    return merged.sort_values("Date").reset_index(drop=True)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r, _ = pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = float("nan")

    signal    = (y_pred > SIGNAL_THR).astype(int)
    actual_up = (y_true > SIGNAL_THR).astype(int)
    n_signals = int(signal.sum())
    sig_acc   = float((signal == actual_up).mean()) if len(actual_up) > 0 else float("nan")

    print(f"   [{name:<20}]  n={len(y_true):>4}  MAE={mae:.5f}  RMSE={rmse:.5f}"
          f"  r={r:.4f}  n_signals={n_signals}  sig_acc={sig_acc:.4f}")
    return dict(model=name, n=int(len(y_true)), mae=mae, rmse=rmse,
                pearson_r=r if not np.isnan(r) else None,
                n_signals=n_signals, signal_acc=sig_acc if not np.isnan(sig_acc) else None)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(tag):
    print(f"\n{'='*64}")
    print(f"  {tag.upper()} — Stacking Regression")
    print(f"{'='*64}")

    val_dfs  = {k: load_val_preds(tag, k)             for k in MODEL_DIRS}
    test_dfs = {k: load_test_preds_with_date(tag, k)  for k in MODEL_DIRS}

    meta_train = merge_on_date(val_dfs)
    meta_test  = merge_on_date(test_dfs)

    if meta_train is None or meta_test is None:
        return None

    pred_cols = [PRED_COL_MAP[k] for k in MODEL_DIRS if PRED_COL_MAP[k] in meta_train.columns
                 and PRED_COL_MAP[k] in meta_test.columns]

    if "y_true" not in meta_train.columns:
        print("   [SKIP] meta-train thiếu y_true.")
        return None
    if "y_true" not in meta_test.columns:
        print("   [SKIP] meta-test thiếu y_true.")
        return None

    if len(meta_train) < 5 or len(pred_cols) < 2:
        print("   [WARN] meta-train quá nhỏ hoặc ít features. Bỏ qua.")
        return None

    X_tr = meta_train[pred_cols].values.astype(np.float32)
    y_tr = meta_train["y_true"].values.astype(np.float32)
    X_te = meta_test[pred_cols].values.astype(np.float32)
    y_te = meta_test["y_true"].values.astype(np.float32)

    print(f"   meta-train={len(meta_train)} (OOF val)  meta-test={len(meta_test)}")
    print(f"   Meta-features ({len(pred_cols)}): {pred_cols}")
    print(f"   Train target mean={y_tr.mean():.4f}  Test target mean={y_te.mean():.4f}")

    # Individual model eval on meta-test
    print(f"\n   Individual base learners on meta-test:")
    base_results = {}
    best_base_mae = np.inf
    for model_key in MODEL_DIRS:
        col = PRED_COL_MAP[model_key]
        if col not in meta_test.columns:
            continue
        p = meta_test[col].values.astype(np.float32)
        r = evaluate(model_key, y_te, p)
        base_results[model_key] = r
        if r["mae"] < best_base_mae:
            best_base_mae = r["mae"]

    # Rank-based Ridge stacking — converts predictions to ranks (0–1) before fitting.
    # LassoCV zeros all coefs when base models have r < 0.3 on small meta-trains.
    # Rank transform removes scale differences; Ridge gives non-zero weights with weak signal.
    def to_rank(arr):
        return rankdata(arr) / len(arr)

    X_tr_r = np.column_stack([to_rank(X_tr[:, i]) for i in range(X_tr.shape[1])])
    X_te_r = np.column_stack([to_rank(X_te[:, i]) for i in range(X_te.shape[1])])

    meta_ridge = Ridge(alpha=0.1)
    meta_ridge.fit(X_tr_r, y_tr)
    pred_tr = meta_ridge.predict(X_tr_r)
    pred_te = meta_ridge.predict(X_te_r)

    coef_str = {k: round(float(v), 4) for k, v in zip(pred_cols, meta_ridge.coef_)}
    print(f"\n   Rank-Ridge alpha=0.1  coefs={coef_str}")
    print(f"\n   Meta-learner (rank-ridge stacking) on meta-test:")
    meta_res = evaluate("stacking_ridge", y_te, pred_te)

    delta = best_base_mae - meta_res["mae"]   # positive = improvement
    sign  = "+" if delta >= 0 else ""
    print(f"\n   ΔMAE vs best individual: {sign}{delta:.5f} (positive = improved)")

    # Save
    model_path  = os.path.join(OUTPUT_DIR, f"ridge_reg_{tag}.joblib")
    pred_path   = os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv")
    result_path = os.path.join(OUTPUT_DIR, f"results_{tag}.json")

    joblib.dump(meta_ridge, model_path)

    out_df = meta_test[["Date"] + [c for c in pred_cols if c in meta_test.columns]
                       + ["y_true"]].copy()
    out_df["y_pred_stack_reg"] = pred_te
    out_df.to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "tag":               tag,
            "n_meta_train":      int(len(meta_train)),
            "n_meta_test":       int(len(meta_test)),
            "pred_cols":         pred_cols,
            "ridge_alpha":       0.1,
            "ridge_coefs":       coef_str,
            "delta_mae_vs_best": float(delta),
            "base_results":      base_results,
            "stacking":          meta_res,
        }, f, indent=2)

    print(f"\n   Saved: {os.path.basename(model_path)}")
    return {**meta_res, "delta_mae_vs_best": float(delta)}


# ─── Entry Point ──────────────────────────────────────────────────────────────

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
        print(f"\n\n{'='*64}")
        print("  SUMMARY — Stacking Regression")
        print(f"{'='*64}")
        print(f"  {'Tag':<36} {'MAE':>9} {'ΔMAE':>9} {'Pearson r':>10}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            name  = tag.replace("integrated_", "")
            mae   = res.get("mae", 0.0)
            delta = res.get("delta_mae_vs_best", 0.0)
            r     = res.get("pearson_r") or 0.0
            sign  = "+" if delta >= 0 else ""
            print(f"  {name:<36}  {mae:>9.5f}  {sign}{delta:>8.5f}  {r:>10.4f}")
        print(f"\n  Predictions saved to: {OUTPUT_DIR}")
    else:
        print("\n  Dam bao da chay 14c-17c regression scripts truoc.")
