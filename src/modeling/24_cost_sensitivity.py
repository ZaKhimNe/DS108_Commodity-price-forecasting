"""
24_cost_sensitivity.py — Transaction Cost Sensitivity Analysis

Kiểm tra kết quả backtesting có còn tốt sau khi trừ phí giao dịch thực tế không.
KC=F round-trip cost: ~0.05%–0.30% per trade (Interactive Brokers → retail).

Output: models/24_cost_sensitivity/
  cost_sensitivity_results.csv
  cost_sensitivity.png

Chạy từ project root:
    python src/modeling/24_cost_sensitivity.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "24_cost_sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_FREE_RATE   = 0.04
PERIODS_PER_YEAR = 52
COST_LEVELS      = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]

# Same dataset configs as 23_bootstrap_sharpe.py
CONFIGS = [
    {
        "tag":         "coffee_daily",
        "pred_csv":    os.path.join(MODELS_DIR, "18_stacking_ensemble",
                                    "test_predictions_integrated_coffee_daily.csv"),
        "integrated":  os.path.join(INTEGRATED_DIR, "integrated_coffee_daily.csv"),
        "signal_col":  "y_pred_stack",
        "model_label": "Coffee Daily Stack",
        "long_short":  False,
        "step":        7,
    },
    {
        "tag":         "coffee_weekly",
        "pred_csv":    os.path.join(MODELS_DIR, "15_rf_baseline",
                                    "test_predictions_integrated_coffee_weekly.csv"),
        "integrated":  os.path.join(INTEGRATED_DIR, "integrated_coffee_weekly.csv"),
        "signal_col":  "y_pred_rf",
        "model_label": "Coffee Weekly Binary RF",
        "long_short":  False,
        "step":        1,
    },
    {
        "tag":         "corn_daily",
        "pred_csv":    os.path.join(MODELS_DIR, "15b_rf_multiclass",
                                    "test_predictions_integrated_corn_daily.csv"),
        "integrated":  os.path.join(INTEGRATED_DIR, "integrated_corn_daily.csv"),
        "signal_col":  "y_pred_rf",
        "model_label": "Corn Daily MC RF",
        "long_short":  True,
        "step":        7,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    excess = returns - RISK_FREE_RATE / PERIODS_PER_YEAR
    return float(excess.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(dd.min())


def total_return(returns: np.ndarray) -> float:
    return float(np.prod(1 + returns) - 1)


def win_rate(returns: np.ndarray, trade_mask: np.ndarray) -> float:
    traded = returns[trade_mask]
    if len(traded) == 0:
        return float("nan")
    return float((traded > 0).mean())


def build_gross_returns(cfg: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (gross_strategy_returns, trade_mask) after non-overlapping step."""
    if not os.path.exists(cfg["pred_csv"]) or not os.path.exists(cfg["integrated"]):
        print(f"  [SKIP] missing files for {cfg['tag']}")
        return None, None

    preds = pd.read_csv(cfg["pred_csv"], parse_dates=["Date"])
    integ = pd.read_csv(cfg["integrated"], parse_dates=["Date"])

    preds["Date"] = pd.to_datetime(preds["Date"]).dt.tz_localize(None)
    integ["Date"] = pd.to_datetime(integ["Date"]).dt.tz_localize(None)

    merged = preds.merge(integ[["Date", "return_future"]], on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.dropna(subset=["return_future"])
    merged = merged.iloc[:: cfg["step"]].reset_index(drop=True)

    if cfg["long_short"]:
        signal_map = {2: 1, 0: -1, 1: 0}
        signals = merged[cfg["signal_col"]].map(signal_map).fillna(0).values.astype(int)
    else:
        signals = merged[cfg["signal_col"]].values.astype(int)

    returns    = merged["return_future"].values.astype(float)
    gross_rets = np.where(signals != 0, signals * returns, 0.0)
    trade_mask = signals != 0
    return gross_rets, trade_mask


def apply_cost(gross_rets: np.ndarray, trade_mask: np.ndarray,
               cost: float) -> np.ndarray:
    """Subtract round-trip cost from each traded period."""
    net = gross_rets.copy()
    net[trade_mask] -= cost
    return net


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Transaction Cost Sensitivity Analysis")
    print("=" * 60)

    all_rows = []

    for cfg in CONFIGS:
        print(f"\n  {cfg['model_label']}")
        gross_rets, trade_mask = build_gross_returns(cfg)
        if gross_rets is None:
            continue

        n_trades = int(trade_mask.sum())
        print(f"    n_periods={len(gross_rets)}, n_trades={n_trades}")
        print(f"    {'Cost':>8}  {'Sharpe':>8}  {'TotalRet':>10}  {'MDD':>8}  {'WinRate':>8}")

        for cost in COST_LEVELS:
            net_rets = apply_cost(gross_rets, trade_mask, cost)
            s  = sharpe(net_rets)
            tr = total_return(net_rets)
            md = max_drawdown(net_rets)
            wr = win_rate(net_rets, trade_mask)

            verdict = (
                "Baseline"   if cost == 0.0 else
                "Robust"     if s >= 1.5   else
                "Acceptable" if s >= 0.8   else
                "Degraded"   if s >= 0.0   else
                "Negative"
            )

            print(f"    {cost*100:>7.2f}%  {s:>8.3f}  {tr:>+10.1%}  {md:>8.3f}  {wr:>8.3f}  {verdict}")

            all_rows.append({
                "tag":          cfg["tag"],
                "model":        cfg["model_label"],
                "n_trades":     n_trades,
                "cost_pct":     round(cost * 100, 3),
                "sharpe":       round(s, 4),
                "total_return": round(tr, 4),
                "mdd":          round(md, 4),
                "win_rate":     round(wr, 4),
                "verdict":      verdict,
            })

    if not all_rows:
        print("\n[WARN] Không có kết quả.")
        return

    results_df = pd.DataFrame(all_rows)

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "cost_sensitivity_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  CSV saved: {out_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = {"Coffee Daily Stack": "#e31a1c",
              "Coffee Weekly Stack": "#4393c3",
              "Corn Daily MC RF":   "#1b7837"}

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    for model_label, grp in results_df.groupby("model"):
        ax.plot(grp["cost_pct"], grp["sharpe"],
                marker="o", markersize=4, lw=1.3,
                color=colors.get(model_label, "#888"),
                label=model_label)

    # Realistic cost band
    ax.axvspan(0.10, 0.30, alpha=0.08, color="#f59e0b",
               label="Realistic range (0.1%–0.3%)")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--", label="Sharpe = 1.0")
    ax.axhline(0.0, color="#ccc", lw=0.6, ls=":")

    ax.set_xlabel("Round-trip Transaction Cost (%)", fontsize=8)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=8)
    ax.set_title("Fig. 7 — Sharpe Sensitivity to Transaction Costs", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.4)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "cost_sensitivity.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
