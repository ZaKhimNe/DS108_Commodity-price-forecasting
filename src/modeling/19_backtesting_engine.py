"""
backtesting_engine.py — Sprint 3: Backtesting & Strategy Evaluation
====================================================================
Convert model predictions thành tín hiệu mua/bán, tính equity curve,
so sánh Sharpe / Max Drawdown / Win Rate với Buy & Hold benchmark.

Dùng để so sánh 3 target formulations (binary / soft / regression)
và 4 learners (LGBM, LSTM, TCN, RF) trên 4 datasets.

Output: models/backtesting/backtest_results_{tag}.csv

Chạy (sau Sprint 4 — tất cả models đã train):
    python src/modeling/backtesting_engine.py
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
    "lgbm":  os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline"),
    "lstm":  os.path.join(PROJECT_ROOT, "models", "16_lstm_hybrid"),
    "tcn":   os.path.join(PROJECT_ROOT, "models", "17_tcn_hybrid"),
    "rf":    os.path.join(PROJECT_ROOT, "models", "15_rf_baseline"),
    "stack": os.path.join(PROJECT_ROOT, "models", "18_stacking_ensemble"),
}
INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "19_backtesting")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

PROB_COL_MAP = {
    "lgbm":  "y_prob_lgb",
    "lstm":  "y_prob_lstm",
    "tcn":   "y_prob_tcn",
    "rf":    "y_prob_rf",
    "stack": "y_prob_stack",
}

RISK_FREE_RATE = 0.04  # annualized (4% USD risk-free)


# ─── Equity curve ─────────────────────────────────────────────────────────────

def compute_equity_curve(
    signals: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Long-only: vào lệnh khi signal=1, nắm giữ 1 period (daily hoặc weekly).
    returns = actual return_future tại mỗi timestep.
    Equity curve bắt đầu từ 1.0.
    """
    strategy_returns = np.where(signals == 1, returns, 0.0)
    equity = np.cumprod(1 + strategy_returns)
    return equity


def sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sharpe Ratio = (mean - rf) / std * sqrt(periods_per_year)."""
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    excess = returns - RISK_FREE_RATE / periods_per_year
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown từ đỉnh."""
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min())


def win_rate(signals: np.ndarray, returns: np.ndarray) -> float:
    """Tỉ lệ trades có lợi nhuận > 0."""
    traded = returns[signals == 1]
    if len(traded) == 0:
        return float("nan")
    return float((traded > 0).mean())


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_predictions(tag: str, model_key: str) -> pd.DataFrame | None:
    """Load test_predictions_{tag}.csv cho một model."""
    path = os.path.join(MODEL_DIRS[model_key], f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df


def load_price_data(tag: str) -> pd.DataFrame | None:
    """Load integrated CSV để lấy return_future thực tế và Date."""
    path = os.path.join(INTEGRATED_DIR, f"{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "return_future"]].dropna()


# ─── Run backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    tag: str,
    model_key: str,
    threshold: float = 0.5,
) -> dict | None:
    """
    Chạy backtest cho một (tag, model) cặp.
    Returns dict metrics hoặc None nếu không đủ data.
    """
    pred_df  = load_predictions(tag, model_key)
    price_df = load_price_data(tag)

    if pred_df is None or price_df is None:
        return None

    prob_col = PROB_COL_MAP[model_key]
    if prob_col not in pred_df.columns:
        return None

    is_daily = "daily" in tag

    # Align theo Date nếu có, không thì dùng cuối cùng N rows của price_df
    if "Date" in pred_df.columns and "Date" in price_df.columns:
        merged = pred_df[["Date", prob_col, "y_true"]].merge(
            price_df, on="Date", how="inner"
        )
    else:
        n = len(pred_df)
        price_tail = price_df.tail(n).reset_index(drop=True)
        merged = pred_df[[prob_col, "y_true"]].copy()
        merged["return_future"] = price_tail["return_future"].values

    if len(merged) < 10:
        return None

    # Daily: return_future = 7-day forward return → các ngày liên tiếp chồng lấn 6/7.
    # Chỉ lấy mỗi 7 hàng (non-overlapping) để tránh thổi phồng equity curve ~7×.
    if is_daily:
        merged = merged.iloc[::7].reset_index(drop=True)
    periods_per_year = 52  # ~52 non-overlapping 7-day periods/year (cả daily lẫn weekly)

    probs   = merged[prob_col].values
    signals = (probs >= threshold).astype(int)
    returns = merged["return_future"].values
    y_true  = merged["y_true"].values

    equity       = compute_equity_curve(signals, returns)
    bh_equity    = compute_equity_curve(np.ones(len(returns)), returns)
    strat_rets   = np.where(signals == 1, returns, 0.0)
    bh_rets      = returns

    n_trades = int(signals.sum())
    n_total  = len(signals)

    return {
        "tag":              tag,
        "model":            model_key,
        "n_test":           n_total,
        "n_trades":         n_trades,
        "trade_rate":       round(n_trades / n_total, 4) if n_total > 0 else 0,
        "sharpe":           round(sharpe_ratio(strat_rets, periods_per_year), 4),
        "max_drawdown":     round(max_drawdown(equity), 4),
        "win_rate":         round(win_rate(signals, returns), 4),
        "total_return":     round(float(equity[-1] - 1), 4),
        "bh_sharpe":        round(sharpe_ratio(bh_rets, periods_per_year), 4),
        "bh_total_return":  round(float(bh_equity[-1] - 1), 4),
        "threshold":        threshold,
    }


def get_threshold(tag: str, model_key: str) -> float:
    """Lấy threshold từ results JSON của model (đã tune trên val)."""
    result_path = os.path.join(MODEL_DIRS[model_key], f"results_{tag}.json")
    if not os.path.exists(result_path):
        return 0.5
    try:
        with open(result_path) as f:
            data = json.load(f)
        return float(data.get("best_threshold", 0.5))
    except Exception:
        return 0.5


# ─── Compare all experiments ──────────────────────────────────────────────────

def compare_all_experiments() -> pd.DataFrame:
    """
    Chạy backtest cho mọi (tag, model) combination.
    Returns DataFrame so sánh tất cả.
    """
    rows = []
    for tag in TAGS:
        for model_key in MODEL_DIRS:
            thr    = get_threshold(tag, model_key)
            result = run_backtest(tag, model_key, threshold=thr)
            if result:
                rows.append(result)

    if not rows:
        print("[WARN] Không có model predictions nào để backtest.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["tag", "sharpe"], ascending=[True, False]).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'═'*72}")
    print("  BACKTEST SUMMARY")
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


def main() -> None:
    print("Chạy backtesting cho tất cả models...")
    results_df = compare_all_experiments()

    if results_df.empty:
        print("Không có kết quả — đảm bảo đã chạy tất cả model scripts trước.")
        return

    print_summary(results_df)

    out_path = os.path.join(OUTPUT_DIR, "backtest_results_all.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n   Saved: {out_path}")

    # Per-tag breakdown
    for tag in TAGS:
        tag_df = results_df[results_df["tag"] == tag]
        if tag_df.empty:
            continue
        tag_path = os.path.join(OUTPUT_DIR, f"backtest_results_{tag}.csv")
        tag_df.to_csv(tag_path, index=False)

    print(f"\n   Decision guide:")
    print(f"   Regression Sharpe >> Binary Sharpe (>0.2)? → chọn regression target")
    print(f"   Else Soft Sharpe >> Binary Sharpe?         → chọn soft target")
    print(f"   Else                                       → nhìn lại CCF lags")


if __name__ == "__main__":
    main()
