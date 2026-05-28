"""
rf_baseline.py — Tầng 2: Random Forest Baseline
================================================
Random Forest tabular learner — base learner thứ 4 trong stacking ensemble.
Mục đích: giảm correlation với LSTM/TCN (kỳ vọng ~0.50 vs ~0.85 hiện tại).

Cùng structure với lgbm_baseline.py:
  - Đọc từ data/integrated/
  - Split 70/10/20 + embargo giống tensor_packing.py
  - Lưu val_predictions_{tag}.csv và test_predictions_{tag}.csv

Chạy:
    python src/modeling/rf_baseline.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "15_rf_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_COL  = "target"
DROP_ALWAYS = {
    "target", "target_binary", "target_soft", "target_reg", "return_future", "Date",
    "target_mc_down", "target_mc_flat", "target_mc_up", "target_multiclass",
}
SPLIT_RATIOS = (0.70, 0.80)

RF_PARAMS = dict(
    n_estimators     = 500,
    max_depth        = 5,
    min_samples_leaf = 20,
    max_features     = 0.5,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
)

RF_PARAMS_WEEKLY = dict(
    n_estimators     = 200,
    max_depth        = 3,
    min_samples_leaf = 30,
    max_features     = 0.4,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
)


# ─── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_split(file_path: str):
    df = pd.read_csv(file_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    horizon    = 7 if "daily" in file_path else 1
    n          = len(df)
    val_start  = int(n * SPLIT_RATIOS[0])
    test_start = int(n * SPLIT_RATIOS[1])

    train_df = df.iloc[: val_start  - horizon].copy()
    val_df   = df.iloc[  val_start  : test_start - horizon].copy()
    test_df  = df.iloc[  test_start :].copy()

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split) < 10:
            raise ValueError(f"Split '{name}' chỉ có {len(split)} rows.")

    return train_df, val_df, test_df


# ─── Feature Preparation ──────────────────────────────────────────────────────

def prepare_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in DROP_ALWAYS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    target = TARGET_COL if TARGET_COL in df.columns else "target_binary"
    y = df[target].values.astype(float)

    return X, y, feature_cols


# ─── Threshold Tuning ─────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_split(name, y_true, y_prob, threshold):
    y_pred   = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    auc_roc  = roc_auc_score(y_true, y_prob)           if has_both else float("nan")
    pr_auc   = average_precision_score(y_true, y_prob) if has_both else float("nan")
    f1       = f1_score(y_true,     y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true,     y_pred, zero_division=0)
    base     = float(y_true.mean())

    print(
        f"   [{name:<6}]  n={len(y_true):>4}  base={base:.3f}  thr={threshold:.2f}"
        f"  AUC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}"
        f"  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
    )

    return {
        "split":      name,
        "n_samples":  int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate":  base,
        "threshold":  float(threshold),
        "auc_roc":    float(auc_roc) if not np.isnan(auc_roc) else None,
        "pr_auc":     float(pr_auc)  if not np.isnan(pr_auc)  else None,
        "f1":         float(f1),
        "precision":  float(prec),
        "recall":     float(rec),
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(file_name: str) -> dict | None:
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n[SKIP] Không tìm thấy: {file_path}")
        return None

    tag      = file_name.replace(".csv", "")
    is_daily = "daily" in file_name

    print(f"\n{'═'*64}")
    print(f"  {tag.upper()} — Random Forest")
    print(f"{'═'*64}")

    train_df, val_df, test_df = load_and_split(file_path)
    print(f"   Rows: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train, y_train, feat_cols = prepare_features(train_df)
    X_val,   y_val,   _         = prepare_features(val_df,   feat_cols)
    X_test,  y_test,  _         = prepare_features(test_df,  feat_cols)

    params  = RF_PARAMS if is_daily else RF_PARAMS_WEEKLY
    spw_str = f"class_weight=balanced ({(y_train == 0).sum()}/{(y_train == 1).sum()} neg/pos)"
    print(f"   Features: {len(feat_cols)}  |  {spw_str}")
    print(f"\n   Training RandomForest  (n={params['n_estimators']}, depth={params['max_depth']})...")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val   = model.predict_proba(X_val)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]

    best_thr = find_optimal_threshold(y_val, prob_val)
    print(f"\n   Optimal threshold (F1 on val): {best_thr:.2f}")

    print()
    results = {
        "train": evaluate_split("TRAIN", y_train, prob_train, best_thr),
        "val":   evaluate_split("VAL",   y_val,   prob_val,   best_thr),
        "test":  evaluate_split("TEST",  y_test,  prob_test,  best_thr),
    }

    # Feature importance
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"\n   Top 10 features:")
    print(imp_df.head(10).to_string(index=False))

    # Save predictions
    val_pred_df  = val_df[["Date"]].copy() if "Date" in val_df.columns else pd.DataFrame(index=range(len(y_val)))
    val_pred_df["y_true"]    = y_val
    val_pred_df["y_prob_rf"] = prob_val

    test_pred_df  = test_df[["Date"]].copy() if "Date" in test_df.columns else pd.DataFrame()
    test_pred_df["y_true"]    = y_test
    test_pred_df["y_prob_rf"] = prob_test
    test_pred_df["y_pred_rf"] = (prob_test >= best_thr).astype(int)

    # Save artifacts
    model_path    = os.path.join(OUTPUT_DIR, f"rf_{tag}.joblib")
    imp_path      = os.path.join(OUTPUT_DIR, f"importance_{tag}.csv")
    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{tag}.csv")
    pred_path     = os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv")
    results_path  = os.path.join(OUTPUT_DIR, f"results_{tag}.json")
    feat_path     = os.path.join(OUTPUT_DIR, f"feature_cols_{tag}.json")

    joblib.dump(model, model_path)
    imp_df.to_csv(imp_path, index=False)
    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(pred_path, index=False)

    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)

    with open(results_path, "w") as f:
        json.dump({
            "file":           file_name,
            "is_daily":       is_daily,
            "n_features":     len(feat_cols),
            "best_threshold": float(best_thr),
            "params":         params,
            "splits":         results,
        }, f, indent=2, default=str)

    print(f"\n   Saved: {os.path.basename(model_path)}")
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    print(f"\n\n{'═'*64}")
    print("  SUMMARY — RF Baseline Test Set")
    print(f"{'═'*64}")
    print(f"  {'File':<42} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1':>7} {'Base':>6}")
    print(f"  {'-'*62}")

    for fname, res in all_results.items():
        t   = res.get("test", {})
        tag = fname.replace("integrated_", "").replace(".csv", "")
        print(
            f"  {tag:<42}"
            f" {t.get('auc_roc', 0.0) or 0.0:>8.4f}"
            f" {t.get('pr_auc',  0.0) or 0.0:>8.4f}"
            f" {t.get('f1',      0.0):>7.4f}"
            f" {t.get('base_rate', 0.0):>6.3f}"
        )

    print(f"\n  Models saved to: {OUTPUT_DIR}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FILES = [
        "integrated_coffee_daily.csv",
        "integrated_coffee_weekly.csv",
        "integrated_corn_daily.csv",
        "integrated_corn_weekly.csv",
    ]

    all_results: dict = {}
    for fname in FILES:
        try:
            res = run_pipeline(fname)
            if res:
                all_results[fname] = res
        except Exception as e:
            print(f"\n[ERROR] {fname}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
    else:
        print("\n[ERROR] Không có file nào được xử lý thành công.")
