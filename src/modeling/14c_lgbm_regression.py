"""
14c_lgbm_regression.py — Tầng 1c: LightGBM Regression
=======================================================
Dự đoán trực tiếp return_future (continuous, clip [-0.30, 0.30]).
Không dùng threshold — signal = (y_pred > 0.05).astype(int) cho backtesting.

Chạy:
    python src/modeling/14c_lgbm_regression.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "14c_lgbm_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_COL  = "target_reg"
SIGNAL_THR  = 0.05   # signal=1 khi y_pred > 5% (dùng ở backtesting)

DROP_ALWAYS = {"target", "target_binary", "target_soft", "target_reg",
               "target_mc_down", "target_mc_flat", "target_mc_up", "target_multiclass",
               "return_future", "Date"}

SPLIT_RATIOS = (0.70, 0.80)

PARAMS_BASE = dict(
    objective         = "regression",
    metric            = ["rmse", "mae"],
    verbose           = -1,
    random_state      = 42,
    n_estimators      = 2000,
    learning_rate     = 0.02,
    num_leaves        = 63,
    max_depth         = 5,
    min_child_samples = 30,
    min_child_weight  = 5,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
)

PARAMS_WEEKLY_OVERRIDE = dict(
    learning_rate     = 0.01,
    num_leaves        = 15,
    max_depth         = 3,
    min_child_samples = 10,
    reg_lambda        = 5.0,
    colsample_bytree  = 0.6,
    n_estimators      = 500,
)


# ─── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_split(file_path):
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
    col = TARGET_COL if TARGET_COL in df.columns else "target"
    y   = df[col].values.astype(float)
    return X, y, feature_cols


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_split(name, y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r, _ = pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = float("nan")

    signal     = (y_pred > SIGNAL_THR).astype(int)
    actual_up  = (y_true > SIGNAL_THR).astype(int)
    n_signals  = int(signal.sum())
    signal_acc = float((signal == actual_up).mean()) if len(actual_up) > 0 else float("nan")

    print(
        f"   [{name:<6}]  n={len(y_true):>4}  MAE={mae:.5f}  RMSE={rmse:.5f}"
        f"  r={r:.4f}  n_signals={n_signals}  signal_acc={signal_acc:.4f}"
    )

    return {
        "split":      name,
        "n_samples":  int(len(y_true)),
        "mae":        mae,
        "rmse":       rmse,
        "pearson_r":  r if not np.isnan(r) else None,
        "n_signals":  n_signals,
        "signal_acc": signal_acc if not np.isnan(signal_acc) else None,
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(file_name):
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n[SKIP] Không tìm thấy: {file_path}")
        return None

    tag      = file_name.replace(".csv", "")
    is_daily = "daily" in file_name

    print(f"\n{'='*64}")
    print(f"  {tag.upper()} — LGBM Regression")
    print(f"{'='*64}")

    train_df, val_df, test_df = load_and_split(file_path)
    print(f"   Rows: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    if TARGET_COL not in train_df.columns:
        print(f"   [SKIP] Thiếu cột {TARGET_COL}. Chạy 11_data_integration.py trước.")
        return None

    X_train, y_train, feat_cols = prepare_features(train_df)
    X_val,   y_val,   _         = prepare_features(val_df,   feat_cols)
    X_test,  y_test,  _         = prepare_features(test_df,  feat_cols)

    print(f"   Features: {len(feat_cols)}")
    print(f"   Target mean — train={y_train.mean():.4f}  val={y_val.mean():.4f}  test={y_test.mean():.4f}")

    params = dict(PARAMS_BASE)
    if not is_daily:
        params.update(PARAMS_WEEKLY_OVERRIDE)

    print(f"\n   Training LGBMRegressor (lr={params['learning_rate']},"
          f" leaves={params['num_leaves']}, max_iter={params['n_estimators']})...")

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    best_iter = getattr(model, "best_iteration_", params["n_estimators"])
    print(f"   Best iteration: {best_iter}")

    pred_train = model.predict(X_train)
    pred_val   = model.predict(X_val)
    pred_test  = model.predict(X_test)

    print()
    results = {
        "train": evaluate_split("TRAIN", y_train, pred_train),
        "val":   evaluate_split("VAL",   y_val,   pred_val),
        "test":  evaluate_split("TEST",  y_test,  pred_test),
    }

    # Feature importance
    imp_df = pd.DataFrame({
        "feature":          feat_cols,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    print(f"\n   Top 10 features (gain):")
    print(imp_df.head(10).to_string(index=False))

    # Val predictions (for stacking 18c)
    val_pred_df = val_df[["Date"]].copy() if "Date" in val_df.columns \
                  else pd.DataFrame(index=range(len(y_val)))
    val_pred_df["y_true"]        = y_val
    val_pred_df["y_pred_lgb_reg"] = pred_val

    # Test predictions
    test_pred_df = test_df[["Date"]].copy() if "Date" in test_df.columns \
                   else pd.DataFrame(index=range(len(y_test)))
    test_pred_df["y_true"]        = y_test
    test_pred_df["y_pred_lgb_reg"] = pred_test

    # Save
    model_path    = os.path.join(OUTPUT_DIR, f"lgbm_reg_{tag}.joblib")
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
            "best_iteration": int(best_iter),
            "signal_thr":     SIGNAL_THR,
            "params":         {k: v for k, v in params.items() if k != "metric"},
            "splits":         results,
        }, f, indent=2, default=str)

    print(f"\n   Saved: {os.path.basename(model_path)}")
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n\n{'='*64}")
    print("  SUMMARY — LGBM Regression Test Set")
    print(f"{'='*64}")
    print(f"  {'File':<38} {'MAE':>8} {'RMSE':>8} {'Pearson r':>10} {'Signals':>8}")
    print(f"  {'-'*62}")
    for fname, res in all_results.items():
        t   = res.get("test", {})
        tag = fname.replace("integrated_", "").replace(".csv", "")
        print(
            f"  {tag:<38}"
            f" {t.get('mae',       0.0):>8.5f}"
            f" {t.get('rmse',      0.0):>8.5f}"
            f" {t.get('pearson_r') or 0.0:>10.4f}"
            f" {t.get('n_signals', 0):>8}"
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

    all_results = {}
    for fname in FILES:
        try:
            res = run_pipeline(fname)
            if res:
                all_results[fname] = res
        except Exception as e:
            print(f"\n[ERROR] {fname}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        print_summary(all_results)
    else:
        print(f"\n[ERROR] Không có file nào xử lý thành công. Kiểm tra: {INPUT_DIR}")
