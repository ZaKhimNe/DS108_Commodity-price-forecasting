"""
19c_backtesting_reg.py — Backtesting Regression
================================================
Signal: go long khi model dự đoán return_future > SIGNAL_THRESHOLD (0.05 = +5%).
Non-overlapping daily: subsample every 7 rows (7-day forward return window).
periods_per_year = 52 cho cả daily lẫn weekly.

Giống 19b_backtesting_mc.py nhưng đọc từ thư mục 14c-18c.
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
    "lgbm":  os.path.join(PROJECT_ROOT, "models", "14c_lgbm_regression"),
    "rf":    os.path.join(PROJECT_ROOT, "models", "15c_rf_regression"),
    "lstm":  os.path.join(PROJECT_ROOT, "models", "16c_lstm_regression"),
    "tcn":   os.path.join(PROJECT_ROOT, "models", "17c_tcn_regression"),
    "stack": os.path.join(PROJECT_ROOT, "models", "18c_stacking_regression"),
}
INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "19c_backtesting_reg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

PRED_COL_MAP = {
    "lgbm":  "y_pred_lgb_reg",
    "rf":    "y_pred_rf_reg",
    "lstm":  "y_pred_lstm_reg",
    "tcn":   "y_pred_tcn_reg",
    "stack": "y_pred_stack_reg",
}

SIGNAL_THRESHOLD = 0.05   # fallback absolute threshold (unused when SIGNAL_PERCENTILE set)
SIGNAL_PERCENTILE = 0.75  # top 25% predictions → long signal (rank-based, avoids scale issue)
RISK_FREE_RATE   = 0.04


# ─── Equity curve ─────────────────────────────────────────────────────────────

def compute_equity_curve(signals, returns):
    strategy_returns = np.where(signals == 1, returns, 0.0)
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
    traded = returns[signals == 1]
    if len(traded) == 0:
        return float("nan")
    return float((traded > 0).mean())


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

    pred_col = PRED_COL_MAP[model_key]
    if pred_col not in pred_df.columns:
        return None

    is_daily = "daily" in tag

    # Align by Date or tail-match
    if "Date" in pred_df.columns and "Date" in price_df.columns:
        cols = ["Date", pred_col] + (["y_true"] if "y_true" in pred_df.columns else [])
        merged = pred_df[cols].merge(price_df, on="Date", how="inner")
    else:
        n          = len(pred_df)
        price_tail = price_df.tail(n).reset_index(drop=True)
        merged     = pred_df[[pred_col]].copy()
        if "y_true" in pred_df.columns:
            merged["y_true"] = pred_df["y_true"].values
        merged["return_future"] = price_tail["return_future"].values

    if len(merged) < 10:
        return None

    # Non-overlapping daily: subsample every 7 rows
    if is_daily:
        merged = merged.iloc[::7].reset_index(drop=True)
    periods_per_year = 52

    # Signal: top SIGNAL_PERCENTILE of predicted returns → long (rank-based avoids scale issue)
    preds    = merged[pred_col].values.astype(np.float32)
    thr      = float(np.percentile(preds, SIGNAL_PERCENTILE * 100))
    signals  = (preds > thr).astype(int)
    returns  = merged["return_future"].values

    equity    = compute_equity_curve(signals, returns)
    bh_equity = compute_equity_curve(np.ones(len(returns)), returns)
    strat_rets = np.where(signals == 1, returns, 0.0)

    n_trades = int(signals.sum())
    n_total  = len(signals)

    return {
        "tag":             tag,
        "model":           model_key,
        "n_test":          n_total,
        "n_trades":        n_trades,
        "trade_rate":      round(n_trades / n_total, 4) if n_total > 0 else 0,
        "sharpe":          round(sharpe_ratio(strat_rets, periods_per_year), 4),
        "max_drawdown":    round(max_drawdown(equity), 4),
        "win_rate":        round(win_rate(signals, returns), 4),
        "total_return":    round(float(equity[-1] - 1), 4),
        "bh_sharpe":       round(sharpe_ratio(returns, periods_per_year), 4),
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
    print(f"\n{'='*72}")
    print("  BACKTEST REGRESSION SUMMARY")
    print(f"{'='*72}")
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
    print("Chay backtesting regression cho tat ca models...")
    results_df = compare_all()

    if results_df.empty:
        print("Khong co ket qua — chay 14c-18c truoc.")
        return

    print_summary(results_df)

    out_path = os.path.join(OUTPUT_DIR, "backtest_results_reg_all.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n   Saved: {out_path}")

    for tag in TAGS:
        tag_df = results_df[results_df["tag"] == tag]
        if tag_df.empty:
            continue
        tag_df.to_csv(os.path.join(OUTPUT_DIR, f"backtest_results_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
