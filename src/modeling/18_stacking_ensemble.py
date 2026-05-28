"""
stacking_ensemble.py — Tầng 3: Stacking Ensemble (LGB + LSTM + TCN)
====================================================================
Combine dự đoán từ ba base learners thành một meta-learner (Logistic Regression).

Prerequisite files (test):
    models/lgbm_baseline/test_predictions_{tag}.csv   (Date, y_true, y_prob_lgb)
    models/lstm_hybrid/test_predictions_{tag}.csv     (y_true, y_prob_lstm — no Date)
    models/tcn_hybrid/test_predictions_{tag}.csv      (y_true, y_prob_tcn — no Date)
    models/lstm_hybrid/results_{tag}.json, models/tcn_hybrid/results_{tag}.json
    data/tensors/win_{W}/{tag}/test_metadata.parquet  (Date alignment for LSTM/TCN)

Prerequisite files (val — OOF meta-train):
    models/lgbm_baseline/val_predictions_{tag}.csv    (Date, y_true, y_prob_lgb)
    models/lstm_hybrid/val_predictions_{tag}.csv      (Date, y_true, y_prob_lstm)
    models/tcn_hybrid/val_predictions_{tag}.csv       (Date, y_true, y_prob_tcn)

OOF Meta-split:
    meta-train = val predictions (OOF — base models chưa thấy khi training)
    meta-test  = full test predictions (held-out hoàn toàn)
    Threshold meta-learner: tuning chỉ trên meta-train (val), apply lên meta-test.

Chạy:
    python src/modeling/stacking_ensemble.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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

LGBM_DIR   = os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline")
LSTM_DIR   = os.path.join(PROJECT_ROOT, "models", "16_lstm_hybrid")
TCN_DIR    = os.path.join(PROJECT_ROOT, "models", "17_tcn_hybrid")
RF_DIR     = os.path.join(PROJECT_ROOT, "models", "15_rf_baseline")
TENSOR_DIR = os.path.join(PROJECT_ROOT, "data",   "tensors")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "18_stacking_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]


# ─── Loading helpers ──────────────────────────────────────────────────────────

def load_lgbm_preds(tag: str) -> pd.DataFrame | None:
    """Load LightGBM test predictions — has Date column directly."""
    path = os.path.join(LGBM_DIR, f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        print(f"   [MISS] LightGBM predictions: {os.path.basename(path)}")
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        print(f"   [WARN] LightGBM pred has no Date column — cannot align")
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df[["Date", "y_true", "y_prob_lgb"]].copy()


def load_deep_preds(tag: str, model_dir: str, prob_col: str) -> pd.DataFrame | None:
    """
    Load LSTM or TCN test predictions (no Date), then attach Date from
    test_metadata.parquet using best_window from results JSON.
    Returns DataFrame[Date, prob_col] or None.
    """
    pred_path = os.path.join(model_dir, f"test_predictions_{tag}.csv")
    if not os.path.exists(pred_path):
        model_name = os.path.basename(model_dir)
        print(f"   [MISS] {model_name} predictions: {os.path.basename(pred_path)}")
        return None

    result_path = os.path.join(model_dir, f"results_{tag}.json")
    if not os.path.exists(result_path):
        print(f"   [MISS] results JSON for {prob_col}: {os.path.basename(result_path)}")
        return None

    with open(result_path) as f:
        results = json.load(f)
    best_w = results.get("best_window")
    if best_w is None:
        print(f"   [WARN] best_window missing in {os.path.basename(result_path)}")
        return None

    meta_path = os.path.join(TENSOR_DIR, f"win_{best_w}", tag, "test_metadata.parquet")
    if not os.path.exists(meta_path):
        print(f"   [MISS] test_metadata.parquet (win={best_w}) for {tag}")
        return None

    pred_df = pd.read_csv(pred_path)
    meta_df = pd.read_parquet(meta_path)

    # Resolve Date from metadata (column or DatetimeIndex)
    if "Date" in meta_df.columns:
        raw_dates = pd.to_datetime(meta_df["Date"].values, utc=True)
    elif isinstance(meta_df.index, pd.DatetimeIndex):
        idx = meta_df.index
        raw_dates = idx.tz_localize("UTC") if idx.tz is None else idx
    else:
        print(f"   [WARN] Cannot find Date in metadata for {tag}/{prob_col}")
        return None

    n_pred, n_meta = len(pred_df), len(raw_dates)
    if n_pred != n_meta:
        print(f"   [WARN] {prob_col}: pred={n_pred} rows, meta={n_meta} rows — trimming to min")
        n = min(n_pred, n_meta)
        pred_df   = pred_df.iloc[-n:].reset_index(drop=True)
        raw_dates = raw_dates[-n:]

    return pd.DataFrame({"Date": raw_dates, prob_col: pred_df[prob_col].values})


# ─── Val prediction loaders (OOF meta-train) ─────────────────────────────────

def load_val_preds(tag: str, model_dir: str, prob_col: str) -> pd.DataFrame | None:
    """
    Load val predictions từ val_predictions_{tag}.csv.
    Các base learner scripts đã gắn Date vào file này.
    Returns DataFrame[Date, y_true?, prob_col] hoặc None.
    """
    path = os.path.join(model_dir, f"val_predictions_{tag}.csv")
    if not os.path.exists(path):
        print(f"   [MISS] val_predictions: {os.path.basename(path)}")
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        print(f"   [WARN] {prob_col} val_predictions has no Date — cannot align")
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    cols = ["Date"] + (["y_true"] if "y_true" in df.columns else []) + [prob_col]
    return df[[c for c in cols if c in df.columns]].copy()


def merge_val_predictions(tag: str) -> pd.DataFrame | None:
    """
    Merge val predictions từ 4 base models theo Date (OOF meta-train).
    y_true lấy từ LGBM val predictions (anchor).
    RF là optional — nếu không có thì dùng 3 models.
    """
    lgbm = load_val_preds(tag, LGBM_DIR, "y_prob_lgb")
    lstm = load_val_preds(tag, LSTM_DIR, "y_prob_lstm")
    tcn  = load_val_preds(tag, TCN_DIR,  "y_prob_tcn")
    rf   = load_val_preds(tag, RF_DIR,   "y_prob_rf")

    if lgbm is None:
        print(f"   [SKIP OOF] LGBM val_predictions required as anchor.")
        return None

    n_available = 1 + (lstm is not None) + (tcn is not None) + (rf is not None)
    if n_available < 2:
        print(f"   [SKIP OOF] Only 1 model val_predictions available.")
        return None

    merged = lgbm.copy()
    if lstm is not None:
        merged = merged.merge(lstm[["Date", "y_prob_lstm"]], on="Date", how="inner")
    if tcn is not None:
        merged = merged.merge(tcn[["Date", "y_prob_tcn"]],  on="Date", how="inner")
    if rf is not None:
        merged = merged.merge(rf[["Date", "y_prob_rf"]],    on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)
    prob_cols = [c for c in ["y_prob_lgb", "y_prob_lstm", "y_prob_tcn", "y_prob_rf"] if c in merged.columns]
    print(f"   OOF meta-train: {len(merged)} rows  |  Models: {prob_cols}")
    return merged


# ─── Merge ────────────────────────────────────────────────────────────────────

def merge_predictions(tag: str) -> pd.DataFrame | None:
    """
    Align and inner-join all available model predictions on Date.
    Requires at least 2 models (including LightGBM as anchor).
    """
    lgbm = load_lgbm_preds(tag)
    lstm = load_deep_preds(tag, LSTM_DIR, "y_prob_lstm")
    tcn  = load_deep_preds(tag, TCN_DIR,  "y_prob_tcn")
    rf   = load_val_preds(tag, RF_DIR, "y_prob_rf")   # RF has Date in test_predictions
    if rf is not None:
        # load_val_preds reads val file — use test file for meta-test
        rf_test_path = os.path.join(RF_DIR, f"test_predictions_{tag}.csv")
        if os.path.exists(rf_test_path):
            rf_df = pd.read_csv(rf_test_path)
            if "Date" in rf_df.columns:
                rf_df["Date"] = pd.to_datetime(rf_df["Date"], utc=True)
                rf = rf_df[["Date", "y_prob_rf"]].copy()
            else:
                rf = None
        else:
            rf = None

    if lgbm is None:
        print(f"   [SKIP] LightGBM required as Date anchor — cannot proceed.")
        return None

    n_available = 1 + (lstm is not None) + (tcn is not None) + (rf is not None)
    if n_available < 2:
        print(f"   [SKIP] Only 1 model available — need at least 2 for stacking.")
        return None

    merged = lgbm.copy()
    if lstm is not None:
        merged = merged.merge(lstm, on="Date", how="inner")
    if tcn is not None:
        merged = merged.merge(tcn, on="Date", how="inner")
    if rf is not None:
        merged = merged.merge(rf,  on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)

    prob_cols = [c for c in ["y_prob_lgb", "y_prob_lstm", "y_prob_tcn", "y_prob_rf"] if c in merged.columns]
    print(f"   Merged: {len(merged)} rows  |  Models: {prob_cols}")
    return merged


# ─── Utilities ────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred   = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    auc_roc  = float(roc_auc_score(y_true, y_prob))           if has_both else float("nan")
    pr_auc   = float(average_precision_score(y_true, y_prob)) if has_both else float("nan")
    f1       = float(f1_score(y_true, y_pred, zero_division=0))
    prec     = float(precision_score(y_true, y_pred, zero_division=0))
    rec      = float(recall_score(y_true, y_pred, zero_division=0))
    base     = float(y_true.mean())

    print(
        f"   [{name:<18}]  n={len(y_true):>4}  base={base:.3f}  thr={threshold:.2f}"
        f"  AUC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}"
    )

    return {
        "model":     name,
        "n":         int(len(y_true)),
        "base_rate": base,
        "threshold": float(threshold),
        "auc_roc":   float(auc_roc) if not np.isnan(auc_roc) else None,
        "pr_auc":    float(pr_auc)  if not np.isnan(pr_auc)  else None,
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(tag: str) -> dict | None:
    print(f"\n{'═'*64}")
    print(f"  {tag.upper()}")
    print(f"{'═'*64}")

    # ── OOF meta-train: val predictions từ base models ──
    meta_train = merge_val_predictions(tag)
    if meta_train is None:
        print(f"   [WARN] Không có val_predictions — fallback về temporal split 50%.")
        merged_full = merge_predictions(tag)
        if merged_full is None:
            return None
        split_idx  = int(len(merged_full) * 0.5)
        meta_train = merged_full.iloc[:split_idx].copy()
        meta_test  = merged_full.iloc[split_idx:].copy()
    else:
        # meta-test = full test predictions
        meta_test = merge_predictions(tag)
        if meta_test is None:
            return None

    prob_cols = [c for c in ["y_prob_lgb", "y_prob_lstm", "y_prob_tcn"] if c in meta_train.columns]
    # Chỉ giữ prob_cols có mặt ở cả meta_train và meta_test
    prob_cols = [c for c in prob_cols if c in meta_test.columns]

    if len(meta_train) < 5:
        print(f"   [WARN] meta-train quá nhỏ ({len(meta_train)} rows) — skipping.")
        return None

    X_tr = meta_train[prob_cols].values.astype(np.float32)
    y_tr = meta_train["y_true"].values.astype(np.float32)
    X_te = meta_test[prob_cols].values.astype(np.float32)
    y_te = meta_test["y_true"].values.astype(np.float32)

    print(
        f"   meta-train={len(meta_train)} (OOF val)  meta-test={len(meta_test)} (full test)"
        f"  base: train={y_tr.mean():.3f}  test={y_te.mean():.3f}"
    )

    # ── Correlation analysis trên meta-test (diagnose before fitting) ──
    if len(prob_cols) > 1:
        corr = pd.DataFrame(meta_test[prob_cols].values, columns=prob_cols).corr()
        print(f"\n   Pairwise correlation of base learner predictions (meta-test):")
        for i, c1 in enumerate(prob_cols):
            for c2 in prob_cols[i + 1:]:
                r    = corr.loc[c1, c2]
                flag = "  ← HIGH — ensemble may not help" if abs(r) > 0.90 else ""
                print(f"     {c1} ↔ {c2}:  r = {r:.3f}{flag}")

    # ── Train meta-learner (Logistic Regression) ──
    if len(np.unique(y_tr)) < 2:
        print(f"\n[SKIP] {tag}: meta-train has only one class — cannot fit LR. "
              f"Corn weekly has 0 positives in val/test due to temporal distribution.")
        return None
    # L2 LogisticRegression — L1 always zeros coefs when base AUC < 0.55 on small meta-trains.
    # L2 gives non-zero weights even with weak signal, enabling ensemble diversification.
    meta_lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    meta_lr.fit(X_tr, y_tr)

    coefs = dict(zip(prob_cols, meta_lr.coef_[0]))
    print(f"\n   L2-LR C=1.0  coefs={', '.join(f'{k}={v:.3f}' for k, v in coefs.items())}")

    prob_meta_tr = meta_lr.predict_proba(X_tr)[:, 1]
    prob_meta_te = meta_lr.predict_proba(X_te)[:, 1]

    # Threshold tuning on meta-train ONLY
    best_thr = find_optimal_threshold(y_tr, prob_meta_tr)

    # ── Base learner comparison on meta-test ──
    # Each model uses its own optimal threshold on meta-test — for display, not model selection
    print(f"\n   Individual base learners on meta-test "
          f"(threshold tuned on meta-test itself — comparison only):")
    base_results: dict = {}
    best_base_auc = 0.0
    for col in prob_cols:
        p         = meta_test[col].values.astype(np.float32)
        thr_base  = find_optimal_threshold(y_te, p)
        r         = evaluate(col, y_te, p, thr_base)
        base_results[col] = r
        best_base_auc = max(best_base_auc, r["auc_roc"] or 0.0)

    # Simple equal-weight average (no training needed)
    prob_avg = np.stack([meta_test[c].values for c in prob_cols], axis=1).mean(axis=1)
    thr_avg  = find_optimal_threshold(y_te, prob_avg)
    avg_res  = evaluate("simple_avg", y_te, prob_avg, thr_avg)

    # ── Meta-learner on meta-test ──
    print(f"\n   Meta-learner (LR stacking) on meta-test  [threshold={best_thr:.2f} from meta-train]:")
    meta_res = evaluate("stacking_lr", y_te, prob_meta_te, best_thr)

    meta_auc = meta_res["auc_roc"] or 0.0
    delta    = meta_auc - best_base_auc
    sign     = "+" if delta >= 0 else ""
    print(f"\n   ΔAUC vs best individual: {sign}{delta:.4f}")

    if delta < 0:
        print(f"   [DIAG] ΔAUC âm — hai nguyên nhân thường gặp:")
        print(f"          1. Predictions tương quan cao → LR không học được weights khác nhau.")
        print(f"          2. Meta-train ({len(meta_train)} rows) quá nhỏ → LR overfit.")
        print(f"          → Xem OOF stacking guide ở cuối file.")

    # ── Save ──
    model_path  = os.path.join(OUTPUT_DIR, f"meta_lr_{tag}.joblib")
    pred_path   = os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv")
    result_path = os.path.join(OUTPUT_DIR, f"results_{tag}.json")

    joblib.dump(meta_lr, model_path)

    out_df = meta_test[["Date"] + prob_cols + ["y_true"]].copy()
    out_df["y_prob_stack"] = prob_meta_te
    out_df["y_pred_stack"] = (prob_meta_te >= best_thr).astype(int)
    out_df.to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "tag":                          tag,
            "meta_train_source":            "oof_val_predictions",
            "n_meta_train":                 int(len(meta_train)),
            "n_meta_test":                  int(len(meta_test)),
            "prob_cols":                    prob_cols,
            "lr_coefs":                     coefs,
            "best_threshold":               float(best_thr),
            "delta_auc_vs_best_individual": float(delta),
            "base_results":                 base_results,
            "avg_blend":                    avg_res,
            "stacking":                     meta_res,
        }, f, indent=2)

    print(f"\n   Saved: {os.path.basename(model_path)}")

    return {**meta_res, "delta_auc_vs_best": float(delta)}


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    print(f"\n\n{'═'*64}")
    print("  SUMMARY — Stacking Ensemble vs Individual Models")
    print(f"{'═'*64}")
    print(f"  {'Tag':<36} {'Stack AUC':>10} {'ΔvsBase':>9} {'F1':>7}")
    print(f"  {'-'*62}")

    for tag, res in all_results.items():
        name  = tag.replace("integrated_", "")
        auc   = res.get("auc_roc")   or 0.0
        delta = res.get("delta_auc_vs_best", 0.0)
        f1    = res.get("f1",        0.0)
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<36}  {auc:>10.4f}  {sign}{delta:>8.4f}  {f1:>7.4f}")

    print(f"\n  Predictions saved to: {OUTPUT_DIR}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results: dict = {}
    for tag in TAGS:
        try:
            res = run_pipeline(tag)
            if res:
                all_results[tag] = res
        except Exception as e:
            print(f"\n[ERROR] {tag}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
    else:
        print("\n[ERROR] Không có tag nào chạy thành công.")
        print(f"        Đảm bảo đã chạy lgbm_baseline.py, lstm_hybrid.py, tcn_hybrid.py trước.")


# ─── OOF Stacking Guide ───────────────────────────────────────────────────────
#
# Phiên bản hiện tại dùng 50% đầu của test set làm meta-train.
# Nhược điểm: meta-train nhỏ, và base models đã thấy toàn bộ val/train khi fit.
#
# OOF (Out-of-Fold) stacking khắc phục bằng cách dùng val predictions làm meta-train:
#   meta-train = val set predictions từ cả 3 models (aligned by Date)
#   meta-test  = test set predictions (như hiện tại)
#   Không cần META_SPLIT nữa — toàn bộ test set là meta-test.
#
# Thay đổi cần thiết trong base learner scripts:
#
# 1. lgbm_baseline.py — thêm val_pred_df vào save block:
#    ─────────────────────────────────────────────────────
#    val_pred_df = val_df[["Date"]].copy()
#    val_pred_df["y_true"]     = y_val
#    val_pred_df["y_prob_lgb"] = prob_val
#    val_pred_df.to_csv(os.path.join(OUTPUT_DIR, f"val_predictions_{tag}.csv"), index=False)
#
# 2. lstm_hybrid.py — trong run_pipeline(), sau eval on val:
#    ─────────────────────────────────────────────────────
#    _, _, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
#    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{file_tag}.csv")
#    pd.DataFrame({"y_true": val_labels, "y_prob_lstm": val_probs}).to_csv(val_pred_path, index=False)
#    # Date từ val_metadata.parquet: data/tensors/win_{W}/{tag}/val_metadata.parquet
#
# 3. tcn_hybrid.py — tương tự lstm_hybrid.py, dùng y_prob_tcn.
#
# 4. stacking_ensemble.py — load_deep_preds dùng val_predictions + val_metadata:
#    ─────────────────────────────────────────────────────────────────────────────
#    Thay "test_predictions" → "val_predictions" và "test_metadata" → "val_metadata"
#    khi build meta-train, giữ nguyên "test_*" cho meta-test.
#
# Khi nào OOF stacking đáng làm:
#   - ΔAUC từ phiên bản này < 0 (meta-learner không học được gì)
#   - Val set có ít nhất 50 samples sau alignment
#   - Ba models có predictions tương quan < 0.95 (nếu > 0.95 thì OOF cũng không giúp)
