"""
22_hurdle_model.py — Two-Stage Hurdle Model
============================================
Giải quyết vấn đề zero-inflation trong return_future của DS108:
  ~65–70% quan sát: |return| ≤ 2.5% (flat zone)
  ~30–35% quan sát:  return > 2.5% (significant upward movement)

Kiến trúc:
  Stage 1 (EXISTING): P(return_future > 2.5%) — Load prob_up từ 14_lgbm_baseline
  Stage 2 (NEW):      E[return_future | return_future > 2.5%] — Regression chỉ trên positive subset
  Combined:           E[return_future] = prob_up × predicted_magnitude

Output dir: models/22_hurdle_model/
  hurdle_{tag}.joblib              — Stage 2 model
  test_predictions_{tag}.csv      — Cột: hurdle_pred, stage2_magnitude, prob_up, y_true
  results_{tag}.json              — r comparison table: single_reg vs stage2_only vs hurdle
  feature_cols_{tag}.json
  zero_inflation_stats_{tag}.json — Thống kê phân phối để vẽ histogram ở Tab 5

Chạy (sau khi đã chạy 14_lgbm_baseline.py và 14c_lgbm_regression.py):
    python src/modeling/22_hurdle_model.py
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

INPUT_DIR       = os.path.join(PROJECT_ROOT, "data", "integrated")
STAGE1_DIR      = os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline")
STAGE1_REG_DIR  = os.path.join(PROJECT_ROOT, "models", "14c_lgbm_regression")
STACK_MC_DIR    = os.path.join(PROJECT_ROOT, "models", "18b_stacking_multiclass")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "models", "22_hurdle_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
THRESHOLD    = 0.025   # 2.5% — khớp với target_threshold trong 11_data_integration.py
SPLIT_RATIOS = (0.70, 0.80)

# Tất cả cột không phải feature — phải khớp với DROP_ALWAYS trong 14c_lgbm_regression.py
DROP_ALWAYS = {
    "target", "target_binary", "target_soft", "target_reg",
    "target_mc_down", "target_mc_flat", "target_mc_up", "target_multiclass",
    "return_future", "Date",
}

# Stage 2 LGBM params — v2: looser regularization cho positive/negative subset
STAGE2_PARAMS_DAILY = dict(
    objective         = "regression",
    metric            = ["rmse", "mae"],
    verbose           = -1,
    random_state      = 42,
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 15,
    max_depth         = 4,
    min_child_samples = 5,
    min_child_weight  = 1,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 0.5,
)

STAGE2_PARAMS_WEEKLY = dict(
    objective         = "regression",
    metric            = ["rmse", "mae"],
    verbose           = -1,
    random_state      = 42,
    n_estimators      = 300,
    learning_rate     = 0.05,
    num_leaves        = 7,
    max_depth         = 3,
    min_child_samples = 3,
    min_child_weight  = 1,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.6,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
)

FILES = [
    "integrated_coffee_daily.csv",
    "integrated_coffee_weekly.csv",
    "integrated_corn_daily.csv",
    "integrated_corn_weekly.csv",
]


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
        if len(split) < 5:
            raise ValueError(f"Split '{name}' chỉ có {len(split)} rows — quá ít.")

    return train_df, val_df, test_df


# ─── Feature Preparation ──────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, feature_cols: list | None = None):
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in DROP_ALWAYS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    X = df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feature_cols


# ─── Stage 1: Load existing binary predictions ────────────────────────────────

def load_stage1_probs(tag: str, split: str):
    """
    Load prob_up từ 14_lgbm_baseline/{val|test}_predictions_{tag}.csv.
    Trả về DataFrame[Date, prob_up] để merge chính xác theo Date.
    """
    path = os.path.join(STAGE1_DIR, f"{split}_predictions_{tag}.csv")
    if not os.path.exists(path):
        print(f"   [MISS] Stage 1 {split} predictions không tìm thấy: {path}")
        return None
    df = pd.read_csv(path)
    if "y_prob_lgb" not in df.columns:
        print(f"   [MISS] Cột y_prob_lgb không có trong {path}")
        return None
    if "Date" not in df.columns:
        print(f"   [WARN] {path} không có cột Date — fallback về array (không merge-safe)")
        return df[["y_prob_lgb"]].rename(columns={"y_prob_lgb": "prob_up"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df[["Date", "y_prob_lgb"]].rename(columns={"y_prob_lgb": "prob_up"})


def load_prob_down(tag: str):
    """
    Load prob_down từ 18b_stacking_multiclass/test_predictions_{tag}.csv.
    Trả về DataFrame[Date, prob_down]. 18b thường ngắn hơn test_df do inner-join 4 models.
    """
    path = os.path.join(STACK_MC_DIR, f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "y_prob_stack_down" not in df.columns:
        return None
    if "Date" not in df.columns:
        print(f"   [WARN] 18b test_predictions không có cột Date — prob_down không dùng được")
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df[["Date", "y_prob_stack_down"]].rename(columns={"y_prob_stack_down": "prob_down"})


def load_stage1_reg_preds(tag: str, split: str):
    """
    Load y_pred_lgb_reg từ 14c_lgbm_regression để so sánh single regression r.
    Trả về DataFrame[Date, single_reg_pred].
    """
    path = os.path.join(STAGE1_REG_DIR, f"{split}_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "y_pred_lgb_reg" not in df.columns:
        return None
    if "Date" not in df.columns:
        return df[["y_pred_lgb_reg"]].rename(columns={"y_pred_lgb_reg": "single_reg_pred"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df[["Date", "y_pred_lgb_reg"]].rename(columns={"y_pred_lgb_reg": "single_reg_pred"})


def _alignment_block(tag, test_df, stage2_magnitude, stage2_neg_magnitude):
    """
    Merge prob_up / prob_down / single_reg vào test_df theo Date (inner join).
    Thay thế tail-slice cũ — đảm bảo y_true[i] = prob_down[i] cho mọi row.
    """
    prob_up_df    = load_stage1_probs(tag, "test")
    prob_down_df  = load_prob_down(tag)
    single_reg_df = load_stage1_reg_preds(tag, "test")

    base_df = test_df.copy()
    base_df["stage2_pos_mag"] = stage2_magnitude
    base_df["stage2_neg_mag"] = stage2_neg_magnitude

    # Merge prob_up theo Date
    if prob_up_df is not None and "Date" in prob_up_df.columns:
        base_df = base_df.merge(prob_up_df, on="Date", how="inner")
        print(f"   After merge prob_up:   n={len(base_df)}")
    elif prob_up_df is not None:
        n = min(len(base_df), len(prob_up_df))
        base_df = base_df.iloc[-n:].copy().reset_index(drop=True)
        base_df["prob_up"] = prob_up_df["prob_up"].values[-n:]
        print(f"   [WARN] prob_up không có Date — tail-slice n={n}")
    else:
        base_df["prob_up"] = base_df["target_binary"].values.astype(float)
        print(f"   [WARN] prob_up không tải được — dùng target_binary làm proxy")

    # Merge prob_down theo Date
    if prob_down_df is not None and "Date" in prob_down_df.columns:
        n_before = len(base_df)
        base_df = base_df.merge(prob_down_df, on="Date", how="inner")
        n_dropped = n_before - len(base_df)
        print(f"   After merge prob_down: n={len(base_df)}"
              f"  (dropped {n_dropped} rows — {n_dropped/max(n_before,1):.1%})")
        if n_dropped / max(n_before, 1) > 0.10:
            print(f"   [WARN] Drop > 10% — kiểm tra Date range 18b predictions")
    else:
        base_df["prob_down"] = np.maximum(1.0 - base_df["prob_up"].values, 0.0)
        print(f"   [WARN] prob_down không tải được — fallback: max(1-prob_up, 0)")

    # Merge single_reg (left join — không drop rows nếu thiếu)
    if single_reg_df is not None and "Date" in single_reg_df.columns:
        base_df = base_df.merge(single_reg_df, on="Date", how="left")

    # Diagnostic — verify date range
    n_align = len(base_df)
    if "Date" in base_df.columns and n_align > 0:
        d_min    = base_df["Date"].min()
        d_max    = base_df["Date"].max()
        orig_max = test_df["Date"].max() if "Date" in test_df.columns else None
        print(f"\n   Alignment diagnostic:")
        print(f"   n_align  = {n_align}")
        print(f"   date_min = {d_min.date() if hasattr(d_min, 'date') else d_min}")
        print(f"   date_max = {d_max.date() if hasattr(d_max, 'date') else d_max}")
        if orig_max is not None and d_max != orig_max:
            print(f"   [WARN] date_max != test_df date_max ({orig_max.date()}) "
                  f"— 18b predictions thiếu ngày cuối test period")
        else:
            print(f"   date_max khớp test_df — alignment OK")

    # Extract arrays
    y_test         = base_df["target_reg"].values
    prob_up_arr    = base_df["prob_up"].values
    prob_down_arr  = base_df["prob_down"].values
    s2_pos         = base_df["stage2_pos_mag"].values
    s2_neg         = base_df["stage2_neg_mag"].values
    single_reg_arr = base_df["single_reg_pred"].values \
                     if "single_reg_pred" in base_df.columns else None

    half_hurdle_pred = prob_up_arr * s2_pos
    full_hurdle_pred = prob_up_arr * s2_pos - prob_down_arr * s2_neg

    return {
        "aligned_df":       base_df,
        "y_test":           y_test,
        "prob_up_test":     prob_up_arr,
        "prob_down_test":   prob_down_arr,
        "stage2_mag":       s2_pos,
        "stage2_neg_mag":   s2_neg,
        "half_hurdle_pred": half_hurdle_pred,
        "full_hurdle_pred": full_hurdle_pred,
        "single_reg_preds": single_reg_arr,
        "n_align":          n_align,
    }


# ─── Zero-Inflation Statistics ────────────────────────────────────────────────

def compute_zero_inflation_stats(df: pd.DataFrame, split_name: str) -> dict:
    """
    Tính thống kê phân phối return_future để vẽ histogram ở Tab 5.
    Trả về dict đủ để Streamlit vẽ mà không cần load lại CSV.
    """
    r = df["return_future"].dropna().values
    n = len(r)
    if n == 0:
        return {}

    flat_mask = np.abs(r) <= THRESHOLD
    pos_mask  = r > THRESHOLD
    neg_mask  = r < -THRESHOLD

    # Histogram bins — dùng fixed edges để 3 splits so sánh được
    bins = np.linspace(-0.30, 0.30, 61)
    counts, edges = np.histogram(r, bins=bins)

    return {
        "split":          split_name,
        "n":              int(n),
        "flat_pct":       float(flat_mask.mean()),
        "pos_pct":        float(pos_mask.mean()),
        "neg_pct":        float(neg_mask.mean()),
        "mean":           float(r.mean()),
        "std":            float(r.std()),
        "median":         float(np.median(r)),
        "p5":             float(np.percentile(r, 5)),
        "p95":            float(np.percentile(r, 95)),
        "threshold":      THRESHOLD,
        # Histogram data — serializable
        "hist_counts":    counts.tolist(),
        "hist_edges":     edges.tolist(),
    }


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def eval_regression(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / Pearson r cho một tập predictions."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r, pval = pearsonr(y_true, y_pred)
        r, pval = float(r), float(pval)
    except Exception:
        r, pval = float("nan"), float("nan")

    print(
        f"   [{name:<28}]  n={len(y_true):>4}  "
        f"MAE={mae:.5f}  RMSE={rmse:.5f}  r={r:.4f}  p={pval:.4f}"
    )
    return {"name": name, "n": int(len(y_true)),
            "mae": mae, "rmse": rmse, "pearson_r": r, "p_value": pval}


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(file_name: str) -> dict | None:
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"\n[SKIP] Không tìm thấy: {file_path}")
        return None

    tag      = file_name.replace(".csv", "")
    is_daily = "daily" in file_name

    print(f"\n{'═'*64}")
    print(f"  {tag.upper()} — Two-Stage Hurdle Model")
    print(f"{'═'*64}")

    # ── 1. Load & split ──
    train_df, val_df, test_df = load_and_split(file_path)
    print(f"   Rows: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Kiểm tra cột cần thiết
    for col in ["target_binary", "target_reg", "return_future"]:
        if col not in train_df.columns:
            print(f"   [SKIP] Thiếu cột '{col}'. Chạy 11_data_integration.py trước.")
            return None

    # ── 2. Zero-inflation statistics ──
    zi_stats = {
        "full":  compute_zero_inflation_stats(
                     pd.concat([train_df, val_df, test_df]), "full"),
        "train": compute_zero_inflation_stats(train_df, "train"),
        "val":   compute_zero_inflation_stats(val_df,   "val"),
        "test":  compute_zero_inflation_stats(test_df,  "test"),
    }
    print(f"\n   Zero-inflation stats (full dataset):")
    fs = zi_stats["full"]
    print(f"   flat={fs['flat_pct']:.1%}  pos={fs['pos_pct']:.1%}  "
          f"neg={fs['neg_pct']:.1%}  threshold=±{THRESHOLD:.1%}")

    # ── 3. Positive subset for Stage 2a ──
    pos_train = train_df[train_df["target_binary"] == 1].copy()
    pos_val   = val_df[val_df["target_binary"] == 1].copy()

    n_pos_train = len(pos_train)
    n_pos_val   = len(pos_val)
    print(f"\n   Positive subset: train={n_pos_train}  val={n_pos_val}")

    if n_pos_train < 20:
        print(f"   [SKIP] Quá ít positive samples ({n_pos_train}) để train Stage 2.")
        return None

    # ── 3b. Negative subset for Stage 2b ──
    neg_mask_train = train_df["return_future"] < -THRESHOLD
    neg_mask_val   = val_df["return_future"]   < -THRESHOLD
    neg_train = train_df[neg_mask_train].copy()
    neg_val   = val_df[neg_mask_val].copy()
    n_neg_train = len(neg_train)
    n_neg_val   = len(neg_val)
    print(f"   Negative subset: train={n_neg_train}  val={n_neg_val}")

    # ── 4. Feature preparation ──
    _, feat_cols = prepare_features(train_df)   # fit feature list trên full train
    print(f"   Features: {len(feat_cols)}")

    X_pos_train, _ = prepare_features(pos_train, feat_cols)
    y_pos_train     = pos_train["target_reg"].values

    X_pos_val, _  = prepare_features(pos_val, feat_cols) if n_pos_val > 0 \
                    else (pd.DataFrame(columns=feat_cols), None)
    y_pos_val     = pos_val["target_reg"].values if n_pos_val > 0 else np.array([])

    # Negative: target = magnitude of down move (positive value)
    X_neg_train, _ = prepare_features(neg_train, feat_cols) if n_neg_train > 0 \
                     else (pd.DataFrame(columns=feat_cols), None)
    y_neg_train     = -neg_train["return_future"].values if n_neg_train > 0 else np.array([])

    X_neg_val, _  = prepare_features(neg_val, feat_cols) if n_neg_val > 0 \
                    else (pd.DataFrame(columns=feat_cols), None)
    y_neg_val     = -neg_val["return_future"].values if n_neg_val > 0 else np.array([])

    X_test, _  = prepare_features(test_df, feat_cols)
    y_test      = test_df["target_reg"].values

    # ── 5. Train Stage 2 ──
    params = STAGE2_PARAMS_DAILY if is_daily else STAGE2_PARAMS_WEEKLY
    print(f"\n   Training Stage 2 LGBMRegressor (positive-only subset)...")
    print(f"   Target mean (pos train): {y_pos_train.mean():.4f}  "
          f"std: {y_pos_train.std():.4f}")

    stage2 = lgb.LGBMRegressor(**params)

    if n_pos_val >= 5:
        stage2.fit(
            X_pos_train, y_pos_train,
            eval_set  = [(X_pos_val, y_pos_val)],
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
    else:
        # Không đủ val positives → train không có early stopping
        stage2.fit(X_pos_train, y_pos_train)
        print(f"   [WARN] Không đủ val positives ({n_pos_val}) → train không có early stopping")

    best_iter = getattr(stage2, "best_iteration_", params["n_estimators"])
    print(f"   Stage 2a best iteration: {best_iter}")

    # ── 5b. Train Stage 2b — E[|return| | return < -threshold] ──
    print(f"\n   Training Stage 2b LGBMRegressor (negative-only subset)...")
    stage2_neg = lgb.LGBMRegressor(**params)
    if n_neg_train >= 20:
        print(f"   Target mean (neg train): {y_neg_train.mean():.4f}  "
              f"std: {y_neg_train.std():.4f}")
        if n_neg_val >= 5:
            stage2_neg.fit(
                X_neg_train, y_neg_train,
                eval_set  = [(X_neg_val, y_neg_val)],
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )
        else:
            stage2_neg.fit(X_neg_train, y_neg_train)
            print(f"   [WARN] Không đủ neg val ({n_neg_val}) → train không có early stopping")
        best_iter_neg = getattr(stage2_neg, "best_iteration_", params["n_estimators"])
        print(f"   Stage 2b best iteration: {best_iter_neg}")
    else:
        print(f"   [WARN] Quá ít neg train ({n_neg_train}) → Stage 2b dùng fallback constant")
        # Fallback: fit với mean target (constant prediction)
        stage2_neg.fit(X_neg_train, y_neg_train) if n_neg_train > 0 \
            else stage2_neg.fit(X_pos_train[:5], np.full(5, THRESHOLD))
        best_iter_neg = 0

    # ── 6. Predict positive magnitude trên toàn bộ test set ──
    stage2_magnitude = stage2.predict(X_test)
    stage2_magnitude = np.maximum(stage2_magnitude, 0.0)  # clip về 0

    # ── 6b. Predict negative magnitude trên toàn bộ test set ──
    stage2_neg_magnitude = stage2_neg.predict(X_test)
    stage2_neg_magnitude = np.maximum(stage2_neg_magnitude, 0.0)  # magnitude luôn dương

    # ── 7+8. Alignment (Date-merge) + Combine predictions ──
    aligned = _alignment_block(tag, test_df, stage2_magnitude, stage2_neg_magnitude)
    test_df_aligned      = aligned["aligned_df"]
    y_test               = aligned["y_test"]
    prob_up_test         = aligned["prob_up_test"]
    prob_down_test       = aligned["prob_down_test"]
    stage2_magnitude     = aligned["stage2_mag"]
    stage2_neg_magnitude = aligned["stage2_neg_mag"]
    half_hurdle_pred     = aligned["half_hurdle_pred"]
    full_hurdle_pred     = aligned["full_hurdle_pred"]
    single_reg_preds_arr = aligned["single_reg_preds"]
    n_align              = aligned["n_align"]

    # ── 9. Evaluate & compare ──
    print(f"\n   Evaluation on test set (n={len(y_test)}):")

    # Stage 2 only — đánh giá trên positive-only subset của test
    pos_test_mask    = test_df_aligned["target_binary"].values == 1
    n_pos_test       = int(pos_test_mask.sum())
    neg_test_mask    = test_df_aligned["return_future"].values < -THRESHOLD
    n_neg_test       = int(neg_test_mask.sum())

    results_compare = {}

    # Full-hurdle combined (toàn bộ test) — v2
    results_compare["full_hurdle_combined"] = eval_regression(
        "Full hurdle combined", y_test, full_hurdle_pred
    )

    # Half-hurdle (v1, giữ cho comparison)
    results_compare["half_hurdle_combined"] = eval_regression(
        "Half hurdle (v1)", y_test, half_hurdle_pred
    )

    # Stage 2a magnitude only — đánh giá trên positive-only subset
    if n_pos_test >= 5:
        results_compare["stage2_positive_only"] = eval_regression(
            f"Stage2a positive-only (n={n_pos_test})",
            y_test[pos_test_mask],
            stage2_magnitude[pos_test_mask],
        )
    else:
        print(f"   [WARN] Chỉ có {n_pos_test} positive rows trong test — bỏ qua stage2_positive_only eval")

    # Stage 2b magnitude on negative subset
    if n_neg_test >= 5:
        # Đánh giá stage2_neg_magnitude trên rows thực sự negative
        # y_true cho negative subset = magnitude of down move = -return_future
        y_neg_true = -y_test[neg_test_mask]
        results_compare["stage2_negative_only"] = eval_regression(
            f"Stage2b negative-only (n={n_neg_test})",
            y_neg_true,
            stage2_neg_magnitude[neg_test_mask],
        )

    # Single regression baseline (14c) — đã được aligned trong _alignment_block
    if single_reg_preds_arr is not None:
        results_compare["single_regression_14c"] = eval_regression(
            "Single regression (14c)", y_test, single_reg_preds_arr
        )

    # r summary table
    fhr = results_compare.get("full_hurdle_combined", {}).get("pearson_r") or 0.0
    hhr = results_compare.get("half_hurdle_combined", {}).get("pearson_r") or 0.0
    sr  = results_compare.get("single_regression_14c", {}).get("pearson_r") or 0.0
    s2  = results_compare.get("stage2_positive_only",  {}).get("pearson_r") or 0.0

    print(f"\n   ── r Comparison Table ──────────────────────────────")
    print(f"   Single regression (14c) r      : {sr:>+.4f}")
    print(f"   Stage 2a positive-only r        : {s2:>+.4f}")
    print(f"   Half hurdle (v1) r              : {hhr:>+.4f}  (positive leg only)")
    print(f"   Full hurdle (v2) r              : {fhr:>+.4f}  (pos − neg legs)")
    delta_fhr_vs_sr  = fhr - sr
    delta_fhr_vs_hhr = fhr - hhr
    delta_s2_vs_sr   = s2 - sr
    print(f"   Δ(full_hurdle vs single)        : {delta_fhr_vs_sr:>+.4f}")
    print(f"   Δ(full_hurdle vs half_hurdle)   : {delta_fhr_vs_hhr:>+.4f}")
    print(f"   Δ(stage2_pos vs single)         : {delta_s2_vs_sr:>+.4f}")

    # Feature importance — Stage 2a (positive)
    imp_df = pd.DataFrame({
        "feature":          feat_cols,
        "importance_gain":  stage2.booster_.feature_importance(importance_type="gain"),
        "importance_split": stage2.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print(f"\n   Top 10 Stage 2a features (gain):")
    print(imp_df.head(10).to_string(index=False))

    # Feature importance — Stage 2b (negative)
    imp_neg_df = pd.DataFrame({
        "feature":          feat_cols,
        "importance_gain":  stage2_neg.booster_.feature_importance(importance_type="gain"),
        "importance_split": stage2_neg.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print(f"\n   Top 10 Stage 2b features (gain):")
    print(imp_neg_df.head(10).to_string(index=False))

    # ── 10. Build output DataFrames ──
    test_pred_df = test_df_aligned[["Date"]].copy() if "Date" in test_df_aligned.columns \
                   else pd.DataFrame(index=range(len(y_test)))
    test_pred_df["y_true"]                = y_test
    test_pred_df["prob_up"]               = prob_up_test
    test_pred_df["prob_down"]             = prob_down_test
    test_pred_df["stage2_magnitude"]      = stage2_magnitude
    test_pred_df["stage2_neg_magnitude"]  = stage2_neg_magnitude
    test_pred_df["half_hurdle_pred"]      = half_hurdle_pred
    test_pred_df["full_hurdle_pred"]      = full_hurdle_pred
    test_pred_df["target_binary"]         = test_df_aligned["target_binary"].values

    # ── 11. Save artifacts ──
    model_path     = os.path.join(OUTPUT_DIR, f"hurdle_{tag}.joblib")
    model_neg_path = os.path.join(OUTPUT_DIR, f"hurdle_neg_{tag}.joblib")
    pred_path      = os.path.join(OUTPUT_DIR, f"test_predictions_{tag}.csv")
    result_path    = os.path.join(OUTPUT_DIR, f"results_{tag}.json")
    feat_path      = os.path.join(OUTPUT_DIR, f"feature_cols_{tag}.json")
    imp_path       = os.path.join(OUTPUT_DIR, f"importance_{tag}.csv")
    imp_neg_path   = os.path.join(OUTPUT_DIR, f"importance_neg_{tag}.csv")
    zi_path        = os.path.join(OUTPUT_DIR, f"zero_inflation_stats_{tag}.json")

    joblib.dump(stage2,     model_path)
    joblib.dump(stage2_neg, model_neg_path)
    test_pred_df.to_csv(pred_path, index=False)
    imp_df.to_csv(imp_path,         index=False)
    imp_neg_df.to_csv(imp_neg_path, index=False)

    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)

    with open(zi_path, "w") as f:
        json.dump(zi_stats, f, indent=2, default=str)

    best_iter_neg_val = getattr(stage2_neg, "best_iteration_", params["n_estimators"])

    results_out = {
        "file":                      file_name,
        "tag":                       tag,
        "is_daily":                  is_daily,
        "threshold":                 THRESHOLD,
        "n_features":                len(feat_cols),
        "stage2a_best_iteration":    int(best_iter),
        "stage2b_best_iteration":    int(best_iter_neg_val),
        "stage2_params":             {k: v for k, v in params.items() if k != "metric"},
        "n_pos_train":               n_pos_train,
        "n_pos_val":                 n_pos_val,
        "n_pos_test":                n_pos_test,
        "n_neg_train":               n_neg_train,
        "n_neg_val":                 n_neg_val,
        "n_neg_test":                n_neg_test,
        "zero_inflation":            {
            "flat_pct_train": round(zi_stats["train"].get("flat_pct", 0), 4),
            "pos_pct_train":  round(zi_stats["train"].get("pos_pct", 0), 4),
            "neg_pct_train":  round(zi_stats["train"].get("neg_pct", 0), 4),
        },
        "r_comparison": {
            "single_regression_14c":          sr,
            "stage2_positive_only":           s2,
            "half_hurdle_combined":           hhr,
            "full_hurdle_combined":           fhr,
            "delta_full_hurdle_vs_single":    round(delta_fhr_vs_sr,  4),
            "delta_full_hurdle_vs_half":      round(delta_fhr_vs_hhr, 4),
            "delta_stage2_vs_single":         round(delta_s2_vs_sr,   4),
        },
        "full_results": results_compare,
    }

    with open(result_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)

    print(f"\n   Saved to {OUTPUT_DIR}")
    return results_out


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    print(f"\n\n{'═'*84}")
    print("  SUMMARY — Two-Stage Full Hurdle Model v2 (r Comparison Table)")
    print(f"{'═'*84}")
    print(f"  {'Dataset':<30} {'Single r':>9} {'Stage2a r':>10} {'Half r (v1)':>12} {'Full r (v2)':>12} {'Δ Full−Half':>12}")
    print(f"  {'-'*86}")
    for tag, res in all_results.items():
        rc   = res.get("r_comparison", {})
        name = tag.replace("integrated_", "").replace(".csv", "")
        sr   = rc.get("single_regression_14c")      or 0.0
        s2   = rc.get("stage2_positive_only")        or 0.0
        hhr  = rc.get("half_hurdle_combined")        or 0.0
        fhr  = rc.get("full_hurdle_combined")        or 0.0
        dlt  = rc.get("delta_full_hurdle_vs_half")   or 0.0
        sign = "+" if dlt >= 0 else ""
        print(f"  {name:<30} {sr:>9.4f} {s2:>10.4f} {hhr:>12.4f} {fhr:>12.4f} {sign}{dlt:>11.4f}")
    print(f"\n  Models saved to: {OUTPUT_DIR}")
    print(f"  hurdle_{{tag}}.joblib         → Stage 2a (positive leg)")
    print(f"  hurdle_neg_{{tag}}.joblib     → Stage 2b (negative leg)")
    print(f"  zero_inflation_stats_{{tag}}.json  → Tab 5 histogram data")
    print(f"  results_{{tag}}.json               → Tab 5 r comparison table")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}
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
        print(f"\n[ERROR] Không có file nào xử lý thành công.")
        print(f"        Đảm bảo đã chạy theo thứ tự:")
        print(f"        1. python src/modeling/14_lgbm_baseline.py")
        print(f"        2. python src/modeling/14c_lgbm_regression.py")
        print(f"        3. python src/modeling/22_hurdle_model.py")
