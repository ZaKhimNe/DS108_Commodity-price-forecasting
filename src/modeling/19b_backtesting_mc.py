"""
19b_backtesting_mc.py — Sprint 3b: Backtesting Multiclass
==========================================================
Chuyển model predictions (3 prob cols) thành tín hiệu:
  signal=1 khi argmax(probs) == 2  (class "up")
  signal=0 otherwise               (long-only strategy)

Giống 19_backtesting_engine.py nhưng đọc từ thư mục 14b-18b.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_DIRS = {
    "lgbm":  os.path.join(PROJECT_ROOT, "models", "14b_lgbm_multiclass"),
    "rf":    os.path.join(PROJECT_ROOT, "models", "15b_rf_multiclass"),
    "lstm":  os.path.join(PROJECT_ROOT, "models", "16b_lstm_multiclass"),
    "tcn":   os.path.join(PROJECT_ROOT, "models", "17b_tcn_multiclass"),
    "stack": os.path.join(PROJECT_ROOT, "models", "18b_stacking_multiclass"),
}
INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "19b_backtesting_mc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

# 3 prob cols per model (class 2 = "up" is the long signal)
PROB_COL_MAP = {
    "lgbm":  ["y_prob_lgb_down",   "y_prob_lgb_flat",   "y_prob_lgb_up"],
    "rf":    ["y_prob_rf_down",    "y_prob_rf_flat",    "y_prob_rf_up"],
    "lstm":  ["y_prob_lstm_down",  "y_prob_lstm_flat",  "y_prob_lstm_up"],
    "tcn":   ["y_prob_tcn_down",   "y_prob_tcn_flat",   "y_prob_tcn_up"],
    "stack": ["y_prob_stack_down", "y_prob_stack_flat", "y_prob_stack_up"],
}

RISK_FREE_RATE = 0.04

# Tags that use Long/Short (instead of Long-Only): go long on "up", short on "down"
LONG_SHORT_TAGS = {"integrated_corn_daily", "integrated_corn_weekly"}

# Probability threshold for signal generation — avoids 0-trade issue when flat dominates.
# Weekly models use lower threshold (0.25) because prob distribution is narrower with fewer observations.
PROB_UP_THR_DAILY    = 0.33
PROB_DOWN_THR_DAILY  = 0.33
PROB_UP_THR_WEEKLY   = 0.25
PROB_DOWN_THR_WEEKLY = 0.25


# ─── Equity curve ─────────────────────────────────────────────────────────────

def compute_equity_curve(signals, returns):
    # signals: +1 long, -1 short, 0 flat. strategy_return = signal * return.
    strategy_returns = np.asarray(signals, dtype=float) * np.asarray(returns, dtype=float)
    return np.cumprod(1 + strategy_returns)


def sharpe_ratio(returns, periods_per_year):
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    excess = returns - RISK_FREE_RATE / periods_per_year
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def win_rate(signals, returns):
    sigs = np.asarray(signals)
    rets = np.asarray(returns)
    active = sigs != 0
    if active.sum() == 0:
        return float("nan")
    # a trade is a win when strategy_return = signal * return > 0
    return float(((sigs[active] * rets[active]) > 0).mean())


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_predictions(tag, model_key):
    path = os.path.join(MODEL_DIRS[model_key], f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df


def load_price_data(tag):
    path = os.path.join(INTEGRATED_DIR, f"{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "return_future"]].dropna()


# ─── Run backtest ─────────────────────────────────────────────────────────────

def run_backtest(tag, model_key):
    pred_df  = load_predictions(tag, model_key)
    price_df = load_price_data(tag)

    if pred_df is None or price_df is None:
        return None

    prob_cols = PROB_COL_MAP[model_key]
    missing   = [c for c in prob_cols if c not in pred_df.columns]
    if missing:
        return None

    is_daily = "daily" in tag
    thr_up   = PROB_UP_THR_DAILY   if is_daily else PROB_UP_THR_WEEKLY
    thr_down = PROB_DOWN_THR_DAILY if is_daily else PROB_DOWN_THR_WEEKLY

    # Align by Date or tail-match
    if "Date" in pred_df.columns and "Date" in price_df.columns:
        merged = pred_df[["Date"] + prob_cols + (["y_true"] if "y_true" in pred_df.columns else [])].merge(
            price_df, on="Date", how="inner"
        )
    else:
        n = len(pred_df)
        price_tail = price_df.tail(n).reset_index(drop=True)
        merged = pred_df[prob_cols].copy()
        if "y_true" in pred_df.columns:
            merged["y_true"] = pred_df["y_true"].values
        merged["return_future"] = price_tail["return_future"].values

    if len(merged) < 10:
        return None

    # Non-overlapping daily: subsample every 7 rows (7-day forward return)
    if is_daily:
        merged = merged.iloc[::7].reset_index(drop=True)
    periods_per_year = 52

    # Signal generation
    probs_mat = merged[prob_cols].values.astype(np.float32)  # (N, 3)
    returns   = merged["return_future"].values
    prob_down = probs_mat[:, 0]
    prob_up   = probs_mat[:, 2]

    if tag in LONG_SHORT_TAGS:
        signals = np.where(prob_up   > thr_up,   1,
                  np.where(prob_down > thr_down, -1, 0))
    else:
        signals = (prob_up > thr_up).astype(int)

    equity    = compute_equity_curve(signals, returns)
    bh_equity = compute_equity_curve(np.ones(len(returns)), returns)
    strat_rets = np.asarray(signals, dtype=float) * returns

    n_trades = int((signals != 0).sum())
    n_total  = len(signals)

    return {
        "tag":          tag,
        "model":        model_key,
        "n_test":       n_total,
        "n_trades":     n_trades,
        "trade_rate":   round(n_trades / n_total, 4) if n_total > 0 else 0,
        "sharpe":       round(sharpe_ratio(strat_rets, periods_per_year), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "win_rate":     round(win_rate(signals, returns), 4),
        "total_return": round(float(equity[-1] - 1), 4),
        "bh_sharpe":    round(sharpe_ratio(returns, periods_per_year), 4),
        "bh_total_return": round(float(bh_equity[-1] - 1), 4),
    }


# ─── Compare all ──────────────────────────────────────────────────────────────

def compare_all():
    rows = []
    for tag in TAGS:
        for model_key in MODEL_DIRS:
            result = run_backtest(tag, model_key)
            if result:
                rows.append(result)

    if not rows:
        print("[WARN] Không có predictions nào để backtest.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["tag", "sharpe"], ascending=[True, False]).reset_index(drop=True)


def print_summary(df):
    print(f"\n{'═'*72}")
    print("  BACKTEST MULTICLASS SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'Tag':<30} {'Model':<8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Return':>8} {'BH Ret':>8}")
    print(f"  {'-'*70}")
    for _, row in df.iterrows():
        tag   = row["tag"].replace("integrated_", "")
        model = row["model"]
        bh    = f"{row['bh_total_return']:>+7.1%}"
        ret   = f"{row['total_return']:>+7.1%}"
        print(
            f"  {tag:<30} {model:<8}"
            f" {row['sharpe']:>8.3f}"
            f" {row['max_drawdown']:>8.3f}"
            f" {row['win_rate']:>8.3f}"
            f" {ret:>8}"
            f" {bh:>8}"
        )


def main():
    print("Chạy backtesting multiclass cho tất cả models...")
    results_df = compare_all()

    if results_df.empty:
        print("Không có kết quả — chạy 14b–18b trước.")
        return

    print_summary(results_df)

    out_path = os.path.join(OUTPUT_DIR, "backtest_results_mc_all.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n   Saved: {out_path}")

    for tag in TAGS:
        tag_df = results_df[results_df["tag"] == tag]
        if tag_df.empty:
            continue
        tag_df.to_csv(os.path.join(OUTPUT_DIR, f"backtest_results_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
