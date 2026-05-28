"""
14b_lgbm_multiclass.py — Tầng 1b: LightGBM Multiclass (down/flat/up)
======================================================================
Giống 14_lgbm_baseline.py nhưng dùng target_multiclass (hard int 0/1/2).
objective="multiclass", num_class=3. Lưu 3 prob columns.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "14b_lgbm_multiclass")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL  = "target_multiclass"   # hard int 0=down, 1=flat, 2=up
CLASS_NAMES = ["down", "flat", "up"]

DROP_ALWAYS = {
    "target", "target_binary", "target_soft", "target_reg",
    "target_mc_down", "target_mc_flat", "target_mc_up", "target_multiclass",
    "return_future", "Date",
}

SPLIT_RATIOS = (0.70, 0.80)

PARAMS_BASE = dict(
    objective          = "multiclass",
    num_class          = 3,
    metric             = ["multi_logloss"],
    verbose            = -1,
    random_state       = 42,
    n_estimators       = 2000,
    learning_rate      = 0.02,
    num_leaves         = 7,
    max_depth          = 4,
    min_child_samples  = 30,
    min_child_weight   = 5,
    subsample          = 0.7,
    subsample_freq     = 1,
    colsample_bytree   = 0.6,
    reg_alpha          = 1.0,
    reg_lambda         = 5.0,
)

PARAMS_WEEKLY_OVERRIDE = dict(
    learning_rate     = 0.01,
    num_leaves        = 4,
    max_depth         = 3,
    min_child_samples = 10,
    reg_lambda        = 10.0,
    colsample_bytree  = 0.5,
    n_estimators      = 500,
)


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


def prepare_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in DROP_ALWAYS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[TARGET_COL].values.astype(int)
    return X, y, feature_cols


def evaluate_split(name, y_true, probs, y_hard):
    """probs: (N, 3), y_hard: (N,) argmax predictions."""
    has_all = len(np.unique(y_true)) == 3
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")) \
              if has_all else float("nan")
    except Exception:
        auc = float("nan")

    f1   = float(f1_score(y_true, y_hard, average="macro", zero_division=0))
    acc  = float(accuracy_score(y_true, y_hard))
    dist = {int(c): int((y_true == c).sum()) for c in sorted(np.unique(y_true))}

    print(
        f"   [{name:<6}]  n={len(y_true):>4}  dist={dist}"
        f"  OvR-AUC={auc:.4f}  MacroF1={f1:.4f}  Acc={acc:.4f}"
    )

    return {
        "split":     name,
        "n_samples": int(len(y_true)),
        "dist":      dist,
        "auc_roc":   auc if not np.isnan(auc) else None,
        "f1":        f1,
        "accuracy":  acc,
    }


def run_pipeline(file_name):
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n[SKIP] {file_path}")
        return None

    tag      = file_name.replace(".csv", "")
    is_daily = "daily" in file_name

    print(f"\n{'═'*64}")
    print(f"  {tag.upper()} — LGBM Multiclass")
    print(f"{'═'*64}")

    train_df, val_df, test_df = load_and_split(file_path)
    print(f"   Rows: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Kiểm tra cột target
    missing = [c for c in [TARGET_COL] if c not in train_df.columns]
    if missing:
        print(f"   [SKIP] Thiếu cột: {missing}. Chạy 11_data_integration.py trước.")
        return None

    X_train, y_train, feat_cols = prepare_features(train_df)
    X_val,   y_val,   _         = prepare_features(val_df,   feat_cols)
    X_test,  y_test,  _         = prepare_features(test_df,  feat_cols)

    params = {**PARAMS_BASE}
    if not is_daily:
        params.update(PARAMS_WEEKLY_OVERRIDE)

    print(f"   Features: {len(feat_cols)}")
    print(f"   Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"\n   Training LightGBM Multiclass (class_weight=balanced)...")

    # Inverse-frequency sample weights so Up/Down aren't drowned by flat majority
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set       = [(X_val, y_val)],
        sample_weight  = sample_weight,
        callbacks      = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    best_iter = getattr(model, "best_iteration_", params["n_estimators"])
    print(f"   Best iteration: {best_iter}")

    probs_train = model.predict_proba(X_train)   # (N, 3)
    probs_val   = model.predict_proba(X_val)
    probs_test  = model.predict_proba(X_test)

    y_hard_train = np.argmax(probs_train, axis=1)
    y_hard_val   = np.argmax(probs_val,   axis=1)
    y_hard_test  = np.argmax(probs_test,  axis=1)

    print()
    results = {
        "train": evaluate_split("TRAIN", y_train, probs_train, y_hard_train),
        "val":   evaluate_split("VAL",   y_val,   probs_val,   y_hard_val),
        "test":  evaluate_split("TEST",  y_test,  probs_test,  y_hard_test),
    }

    # Feature importance
    imp_df = pd.DataFrame({
        "feature":          feat_cols,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print(f"\n   Top 10 features (gain):")
    print(imp_df.head(10).to_string(index=False))

    # Val predictions (cho OOF stacking)
    val_pred_df = val_df[["Date"]].copy() if "Date" in val_df.columns \
                  else pd.DataFrame(index=range(len(y_val)))
    val_pred_df["y_true"] = y_val
    for i, cls in enumerate(CLASS_NAMES):
        val_pred_df[f"y_prob_lgb_{cls}"] = probs_val[:, i]

    # Test predictions
    test_pred_df = test_df[["Date"]].copy() if "Date" in test_df.columns \
                   else pd.DataFrame(index=range(len(y_test)))
    test_pred_df["y_true"]      = y_test
    test_pred_df["y_pred_lgb"]  = y_hard_test
    for i, cls in enumerate(CLASS_NAMES):
        test_pred_df[f"y_prob_lgb_{cls}"] = probs_test[:, i]

    # Save
    joblib.dump(model,   os.path.join(OUTPUT_DIR, f"lgbm_{tag}.joblib"))
    imp_df.to_csv(       os.path.join(OUTPUT_DIR, f"importance_{tag}.csv"), index=False)
    val_pred_df.to_csv(  os.path.join(OUTPUT_DIR, f"val_predictions_{tag}.csv"), index=False)
    test_pred_df.to_csv( os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, f"feature_cols_{tag}.json"), "w") as f:
        json.dump(feat_cols, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, f"results_{tag}.json"), "w") as f:
        json.dump({
            "file":           file_name,
            "is_daily":       is_daily,
            "n_features":     len(feat_cols),
            "best_iteration": int(best_iter),
            "params":         {k: v for k, v in params.items() if k != "metric"},
            "splits":         results,
        }, f, indent=2, default=str)

    print(f"\n   Saved to {OUTPUT_DIR}")
    return results


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
        print(f"\n\n{'═'*64}")
        print("  SUMMARY — LGBM Multiclass Test Set")
        print(f"{'═'*64}")
        for fname, res in all_results.items():
            t   = res.get("test", {})
            tag = fname.replace("integrated_", "").replace(".csv", "")
            print(f"  {tag:<42}  OvR-AUC={t.get('auc_roc') or 0.0:.4f}"
                  f"  F1-macro={t.get('f1', 0.0):.4f}  Acc={t.get('accuracy', 0.0):.4f}")
        print(f"\n  Models saved to: {OUTPUT_DIR}")
