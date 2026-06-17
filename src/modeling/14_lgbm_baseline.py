"""
lgbm_baseline.py — Tầng 1: LightGBM Tabular Baseline
======================================================
Đọc từ data/integrated/, split 70/10/20 + embargo (giống tensor_packing.py),
dùng snapshot tại T (hàng cuối window) thay vì 3D tensor.

Đây là benchmark mạnh nhất cho tabular tài chính — model phức tạp hơn
(LSTM, TCN, TFT) phải thắng baseline này để được coi là có giá trị.

Chạy:
    python src/modeling/lgbm_baseline.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
# TARGET_MODE: "binary" | "soft" | "regression"
TARGET_MODE    = "binary"
TARGET_COL_MAP = {"binary": "target_binary", "soft": "target_soft", "regression": "target_reg"}
TARGET_COL     = TARGET_COL_MAP.get(TARGET_MODE, "target_binary")

DROP_ALWAYS = {
    "target", "target_binary", "target_soft", "target_reg", "return_future", "Date",
    "target_mc_down", "target_mc_flat", "target_mc_up", "target_multiclass",
}

SPLIT_RATIOS = (0.70, 0.80)  # val_start=70%, test_start=80%

# P1: rolling lookback — train only on most recent N rows to reduce regime shift
# 1000 daily ≈ 4 years; None = use all train data (weekly sets are small)
TRAIN_LOOKBACK_ROWS = {"daily": 1000, "weekly": None}

# Walkforward CV — each test chunk is predicted by a model trained on the preceding
# WALKFORWARD_TRAIN_ROWS rows. Addresses regime shift (best_iter=1 on fixed split).
# Set WALKFORWARD_DAILY=False to revert to the old fixed-split behaviour.
WALKFORWARD_DAILY      = True   # rolling train window for daily datasets
WALKFORWARD_TRAIN_ROWS = 1000    # Tăng kích thước mẫu nền
WALKFORWARD_STEP_ROWS  = 130   # retrain every ~6 months

# LightGBM params cơ bản — override theo daily/weekly ở dưới
PARAMS_BASE = dict(
    objective          = "binary",
    metric             = ["binary_logloss", "auc"],
    verbose            = -1,
    random_state       = 42,
    n_estimators       = 2000,
    learning_rate      = 0.015,       # Giảm tốc độ học để tăng tính ổn định
    num_leaves         = 6,           # Hạ bớt 1 lá để khống chế độ sâu cấu trúc
    max_depth          = 4,
    min_child_samples  = 35,          # Hạ từ 50 xuống 35 để phù hợp với phân bổ mẫu
    min_child_weight   = 5,
    min_split_gain     = 0.02,        # Hạ ngưỡng gain để cây có thể phân nhánh mịn
    subsample          = 0.7,
    subsample_freq     = 1,
    colsample_bytree   = 0.6,
    reg_alpha          = 1.0,
    reg_lambda         = 10.0,        # Đưa về mức 10.0 để giảm áp lực co hẹp trọng số
)
# Weekly có ít sample hơn → giảm complexity mạnh hơn để tránh overfit
PARAMS_WEEKLY_OVERRIDE = dict(
    learning_rate     = 0.01,
    num_leaves        = 4,
    max_depth         = 3,
    min_child_samples = 10,
    reg_lambda        = 10.0,
    colsample_bytree  = 0.5,
    n_estimators      = 500,
)


# ─── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_split(
    file_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Load integrated CSV, sort theo Date, split 70/10/20 với embargo gap.
    Embargo = horizon rows (tránh target leakage tại biên split).

    Returns: train_df, val_df, test_df, full_df (pre-embargo), test_start_idx
    full_df and test_start_idx are used by run_walkforward_test().
    """
    df = pd.read_csv(file_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    # Embargo: số hàng cần bỏ ra tại biên để tránh target overlap
    horizon    = 7 if "daily" in file_path else 1
    n          = len(df)
    val_start  = int(n * SPLIT_RATIOS[0])
    test_start = int(n * SPLIT_RATIOS[1])

    train_df = df.iloc[: val_start  - horizon].copy()
    val_df   = df.iloc[  val_start  : test_start - horizon].copy()
    test_df  = df.iloc[  test_start :].copy()

    # Rolling lookback: truncate train to most recent N rows (reduces regime shift)
    freq_key = "daily" if "daily" in file_path else "weekly"
    lookback  = TRAIN_LOOKBACK_ROWS.get(freq_key)
    if lookback and len(train_df) > lookback:
        train_df = train_df.iloc[-lookback:].copy()
        print(f"   [Rolling] Train truncated to last {lookback} rows")

    # Kiểm tra minimum samples — weekly data có thể rất ít
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split) < 10:
            raise ValueError(f"Split '{name}' chỉ có {len(split)} rows — không đủ để train.")

    return train_df, val_df, test_df, df, test_start


# ─── Feature Preparation ──────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Tách features và target.
    Nếu feature_cols=None (train), tự động detect từ df.
    Nếu feature_cols được truyền (val/test), align về cùng cột.
    """
    # Xác định cột features
    if feature_cols is None:
        drop_set = DROP_ALWAYS | {TARGET_COL}
        feature_cols = [
            c for c in df.columns
            if c not in drop_set
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    # Lọc các cột thực sự tồn tại (val/test có thể thiếu một số cột)
    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        print(f"   [WARN] {len(missing)} feature cols không có trong split — điền 0: {list(missing)[:5]}...")

    X = df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Fallback: nếu target_binary chưa có (dữ liệu cũ), dùng 'target'
    col = TARGET_COL if TARGET_COL in df.columns else "target"
    y = df[col].values.astype(float)

    return X, y, feature_cols


# ─── Class Imbalance ──────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    scale_pos_weight = #negative / #positive.
    LightGBM nhân loss của positive class lên để compensate imbalance.
    Cap tại 10 để tránh training không ổn định khi base rate rất thấp.
    """
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return float(min(neg / pos, 10.0))


# ─── Threshold Tuning ─────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Tìm threshold tối ưu trên validation set.
    metric = 'f1' (default) hoặc 'f1_weighted'.

    Note: Threshold tuning PHẢI chỉ dùng val set.
    Apply threshold tìm được lên test set mà KHÔNG retune.
    """
    thresholds  = np.arange(0.05, 0.95, 0.01)
    best_score  = -1.0
    best_thr    = 0.5

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score  = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr   = thr

    return float(best_thr)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Tính đầy đủ metrics cho một split và in ra console."""
    y_pred = (y_prob >= threshold).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1
    auc_roc = roc_auc_score(y_true, y_prob)          if has_both_classes else float("nan")
    pr_auc  = average_precision_score(y_true, y_prob) if has_both_classes else float("nan")
    f1      = f1_score(y_true,    y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true,    y_pred, zero_division=0)

    base_rate = float(y_true.mean())
    lift      = (prec / base_rate) if base_rate > 0 else float("nan")

    print(
        f"   [{name:<6}]  n={len(y_true):>4}  base={base_rate:.3f}  thr={threshold:.2f}"
        f"  AUC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}"
        f"  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  Lift={lift:.2f}x"
    )

    return {
        "split":      name,
        "n_samples":  int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate":  base_rate,
        "threshold":  float(threshold),
        "auc_roc":    float(auc_roc)  if not np.isnan(auc_roc) else None,
        "pr_auc":     float(pr_auc)   if not np.isnan(pr_auc)  else None,
        "f1":         float(f1),
        "precision":  float(prec),
        "recall":     float(rec),
        "lift":       float(lift)     if not np.isnan(lift)    else None,
    }


# ─── Walkforward Rolling Window ───────────────────────────────────────────────

def run_walkforward_test(
    full_df: pd.DataFrame,
    test_start: int,
    feature_cols: list[str],
    params_base: dict,
) -> tuple[np.ndarray, np.ndarray]:
    n          = len(full_df)
    all_probs: list[np.ndarray] = []
    all_true:  list[np.ndarray] = []
    n_windows  = 0

    for start in range(test_start, n, WALKFORWARD_STEP_ROWS):
        end       = min(start + WALKFORWARD_STEP_ROWS, n)
        train_s   = max(0, start - WALKFORWARD_TRAIN_ROWS)

        win_train = full_df.iloc[train_s:start]
        win_test  = full_df.iloc[start:end]

        X_tr_full, y_tr_full, _ = prepare_features(win_train, feature_cols)
        X_te,      y_te,      _ = prepare_features(win_test,  feature_cols)

        # Tách 15% cuối window làm val nội bộ cho early stopping
        n_val_wf = max(30, int(len(y_tr_full) * 0.15))
        X_tr     = X_tr_full.iloc[:-n_val_wf]
        y_tr     = y_tr_full[:-n_val_wf]
        X_wv     = X_tr_full.iloc[-n_val_wf:]
        y_wv     = y_tr_full[-n_val_wf:]

        spw = compute_scale_pos_weight(y_tr)
        p   = {**params_base, "scale_pos_weight": spw}

        m = lgb.LGBMClassifier(**p)
        m.fit(
            X_tr, y_tr,
            eval_set  = [(X_wv, y_wv)],
            callbacks = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        bi = getattr(m, "best_iteration_", p["n_estimators"])
        pos_rate = float(y_tr.mean())
        print(f"   [WF] rows=[{start}:{end}]  train=[{train_s}:{start}]  best_iter={bi}  pos={pos_rate:.3f}")

        all_probs.append(m.predict_proba(X_te)[:, 1])
        all_true.append(y_te)
        n_windows += 1

    total = sum(len(p) for p in all_probs)
    print(f"   Walkforward: {n_windows} windows → {total} test rows")
    return np.concatenate(all_true), np.concatenate(all_probs)

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(file_name: str) -> dict | None:
    """
    Full training pipeline cho một file integrated.
    Returns: dict kết quả hoặc None nếu file không tồn tại.
    """
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n[SKIP] Không tìm thấy: {file_path}")
        return None

    tag      = file_name.replace(".csv", "")
    is_daily = "daily" in file_name
    use_wf   = WALKFORWARD_DAILY and is_daily

    print(f"\n{'═'*64}")
    print(f"  {tag.upper()}{'  [WALKFORWARD]' if use_wf else ''}")
    print(f"{'═'*64}")

    # 1. Load & split
    train_df, val_df, test_df, full_df, test_start_idx = load_and_split(file_path)
    print(f"   Rows: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # 2. Features — fit trên train, apply cho val/test
    X_train, y_train, feat_cols = prepare_features(train_df)

    missing_val  = [c for c in feat_cols if c not in val_df.columns]
    missing_test = [c for c in feat_cols if c not in test_df.columns]
    if missing_val or missing_test:
        raise AssertionError(
            f"Schema mismatch in {file_name} — all splits must come from the same integrated CSV.\n"
            f"  val missing:  {missing_val}\n"
            f"  test missing: {missing_test}"
        )

    X_val,   y_val,   _         = prepare_features(val_df,   feat_cols)
    X_test,  y_test,  _         = prepare_features(test_df,  feat_cols)

    n_feat   = len(feat_cols)
    spw      = compute_scale_pos_weight(y_train)
    base_str = (
        f"train={y_train.mean():.3f}  "
        f"val={y_val.mean():.3f}  "
        f"test={y_test.mean():.3f}"
    )
    print(f"   Features: {n_feat}  |  Base rate: {base_str}")
    print(f"   scale_pos_weight = {spw:.2f}")

    # 3. Build params — override cho weekly
    # params_base: no scale_pos_weight (added per-model in walkforward)
    params_base = {**PARAMS_BASE}
    if not is_daily:
        params_base.update(PARAMS_WEEKLY_OVERRIDE)

    params = {**params_base, "scale_pos_weight": spw}

    # 4. Train single model on train_df (used for: val evaluation, threshold tuning,
    #    train-set metrics, feature importance, model artifact save)
    print(f"\n   Training LightGBM  (lr={params['learning_rate']},"
          f" leaves={params['num_leaves']}, max_iter={params['n_estimators']})...")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set    = [(X_val, y_val)],
        callbacks   = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    best_iter = getattr(model, "best_iteration_", params["n_estimators"])
    print(f"   Best iteration (single model / val set): {best_iter}")

    # 5. Predict probabilities
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val   = model.predict_proba(X_val)[:, 1]

    # Test predictions: walkforward (daily) or single-model (weekly / disabled)
    if use_wf:
        print(f"\n   Walkforward test predictions"
              f" (train_rows={WALKFORWARD_TRAIN_ROWS}, step={WALKFORWARD_STEP_ROWS})...")
        y_test, prob_test = run_walkforward_test(
            full_df, test_start_idx, feat_cols, params_base
        )
    else:
        prob_test = model.predict_proba(X_test)[:, 1]

    # 6. Threshold tuning trên VAL — KHÔNG nhìn test
    best_thr = find_optimal_threshold(y_val, prob_val)
    print(f"\n   Optimal threshold (F1 on val): {best_thr:.2f}")

    # 7. Evaluate
    print()
    results = {
        "train": evaluate_split("TRAIN", y_train, prob_train, best_thr),
        "val":   evaluate_split("VAL",   y_val,   prob_val,   best_thr),
        "test":  evaluate_split("TEST",  y_test,  prob_test,  best_thr),
    }

    # Kiểm tra overfit signal
    train_auc = results["train"]["auc_roc"] or 0
    val_auc   = results["val"]["auc_roc"]   or 0
    gap       = train_auc - val_auc
    if gap > 0.10:
        print(f"\n   [WARN] Overfit gap lớn: train={train_auc:.4f} - val={val_auc:.4f} = {gap:.4f}")
        print(f"          → Tăng reg_alpha/reg_lambda hoặc giảm num_leaves.")

    # 8. Feature importance
    imp_df = pd.DataFrame({
        "feature":          feat_cols,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print(f"\n   Top 15 features (gain-based importance):")
    print(imp_df.head(15).to_string(index=False))

    # 9. Save predictions (cho stacking ensemble sau này)
    val_pred_df = val_df[["Date"]].copy() if "Date" in val_df.columns else pd.DataFrame(index=range(len(y_val)))
    val_pred_df["y_true"]     = y_val
    val_pred_df["y_prob_lgb"] = prob_val

    test_pred_df = test_df[["Date"]].copy() if "Date" in test_df.columns else pd.DataFrame()
    test_pred_df["y_true"]     = y_test
    test_pred_df["y_prob_lgb"] = prob_test
    test_pred_df["y_pred_lgb"] = (prob_test >= best_thr).astype(int)

    # 10. Save artifacts
    model_path    = os.path.join(OUTPUT_DIR, f"lgbm_{tag}.joblib")
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
            "file":                file_name,
            "is_daily":            is_daily,
            "walkforward":         use_wf,
            "walkforward_train_rows": WALKFORWARD_TRAIN_ROWS if use_wf else None,
            "walkforward_step_rows":  WALKFORWARD_STEP_ROWS  if use_wf else None,
            "n_features":          n_feat,
            "best_iteration":      int(best_iter),
            "best_threshold":      float(best_thr),
            "scale_pos_weight":    float(spw),
            "params":              {k: v for k, v in params.items() if k != "metric"},
            "splits":              results,
        }, f, indent=2, default=str)

    print(f"\n   Saved: {os.path.basename(model_path)}")
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    """In bảng tóm tắt AUC và F1 trên test set cho tất cả files."""
    print(f"\n\n{'═'*64}")
    print("  SUMMARY — Test Set Performance")
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
    print("  Dùng test_predictions_*.csv cho stacking ensemble (Tầng 3).")


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
        print(f"        Kiểm tra thư mục: {INPUT_DIR}")
