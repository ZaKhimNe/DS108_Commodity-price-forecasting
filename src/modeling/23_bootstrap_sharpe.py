"""
23_bootstrap_sharpe.py — Bootstrap CI cho Sharpe Ratio

Trả lời câu hỏi của reviewer: "Sharpe = 2.154 từ 48 trades có đáng tin không?"
Bootstrap 10,000 lần → 95% CI, P(Sharpe>0), P(Sharpe>1).

Output: models/23_bootstrap_sharpe/
  bootstrap_results.csv
  sharpe_ci_coffee_daily.png
  sharpe_ci_coffee_weekly.png
  sharpe_ci_corn_daily.png

Chạy từ project root:
    python src/modeling/23_bootstrap_sharpe.py
"""

import os
import json
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
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "23_bootstrap_sharpe")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_FREE_RATE = 0.04   # 4% annualized
N_BOOTSTRAP    = 10_000
PERIODS_PER_YEAR = 52   # non-overlapping 7-day periods per year

# ─── Datasets to analyse ──────────────────────────────────────────────────────
# (pred_csv relative to MODELS_DIR, integrated_csv relative to INTEGRATED_DIR,
#  signal_col, prob_col_or_none, model_label, is_long_short, step)
CONFIGS = [
    {
        "tag":          "coffee_daily",
        "pred_csv":     os.path.join(MODELS_DIR, "18_stacking_ensemble",
                                     "test_predictions_integrated_coffee_daily.csv"),
        "integrated":   os.path.join(INTEGRATED_DIR, "integrated_coffee_daily.csv"),
        "signal_col":   "y_pred_stack",
        "prob_col":     "y_prob_stack",
        "model_label":  "Stack",
        "long_short":   False,
        "step":         7,
    },
    {
        "tag":          "coffee_weekly",
        "pred_csv":     os.path.join(MODELS_DIR, "15_rf_baseline",
                                     "test_predictions_integrated_coffee_weekly.csv"),
        "integrated":   os.path.join(INTEGRATED_DIR, "integrated_coffee_weekly.csv"),
        "signal_col":   "y_pred_rf",
        "prob_col":     "y_prob_rf",
        "model_label":  "Binary RF",
        "long_short":   False,
        "step":         1,
    },
    {
        "tag":          "corn_daily",
        "pred_csv":     os.path.join(MODELS_DIR, "15b_rf_multiclass",
                                     "test_predictions_integrated_corn_daily.csv"),
        "integrated":   os.path.join(INTEGRATED_DIR, "integrated_corn_daily.csv"),
        "signal_col":   "y_pred_rf",
        "prob_col":     None,
        "model_label":  "MC RF L/S",
        "long_short":   True,
        "step":         7,
    },
]


# ─── Core helpers ─────────────────────────────────────────────────────────────

def sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe. Mirrors formula in 19_backtesting_engine.py."""
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    excess = returns - RISK_FREE_RATE / PERIODS_PER_YEAR
    return float(excess.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


def build_strategy_returns(cfg: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Merge predictions + integrated CSV, apply non-overlapping step,
    return (strategy_returns, signals) arrays.
    """
    if not os.path.exists(cfg["pred_csv"]):
        print(f"  [SKIP] pred not found: {cfg['pred_csv']}")
        return None, None
    if not os.path.exists(cfg["integrated"]):
        print(f"  [SKIP] integrated not found: {cfg['integrated']}")
        return None, None

    preds = pd.read_csv(cfg["pred_csv"], parse_dates=["Date"])
    integ = pd.read_csv(cfg["integrated"], parse_dates=["Date"])

    preds["Date"] = pd.to_datetime(preds["Date"]).dt.tz_localize(None)
    integ["Date"] = pd.to_datetime(integ["Date"]).dt.tz_localize(None)

    merged = preds.merge(integ[["Date", "return_future"]], on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.dropna(subset=["return_future"])

    # Non-overlapping sampling (mirrors 19_backtesting_engine.py)
    merged = merged.iloc[:: cfg["step"]].reset_index(drop=True)

    if cfg["long_short"]:
        # MC RF: 2=Up→long(+1), 0=Down→short(-1), 1=Flat→0
        signal_map = {2: 1, 0: -1, 1: 0}
        signals = merged[cfg["signal_col"]].map(signal_map).fillna(0).values.astype(int)
    else:
        signals = merged[cfg["signal_col"]].values.astype(int)

    returns = merged["return_future"].values.astype(float)
    strat_rets = np.where(signals != 0, signals * returns, 0.0)
    return strat_rets, signals


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_sharpe(strategy_returns: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     seed: int = 42) -> dict:
    """
    Bootstrap CI for Sharpe ratio.
    Returns dict with obs/ci/median/p_positive/p_above_1.
    """
    rng = np.random.default_rng(seed)
    n = len(strategy_returns)
    sharpe_obs = sharpe(strategy_returns)

    bs_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(strategy_returns, size=n, replace=True)
        s = sharpe(sample)
        if not np.isnan(s):
            bs_sharpes.append(s)

    bs = np.array(bs_sharpes)
    ci_lower    = float(np.percentile(bs, 2.5))
    ci_upper    = float(np.percentile(bs, 97.5))
    median_bs   = float(np.percentile(bs, 50))
    p_positive  = float(np.mean(bs > 0))
    p_above_1   = float(np.mean(bs > 1))

    return {
        "sharpe_obs":  round(sharpe_obs, 4),
        "ci_lower":    round(ci_lower, 4),
        "ci_upper":    round(ci_upper, 4),
        "median_bs":   round(median_bs, 4),
        "p_positive":  round(p_positive, 4),
        "p_above_1":   round(p_above_1, 4),
        "n_valid_bs":  len(bs),
        "_bs_array":   bs,   # for plotting, not saved to CSV
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_bootstrap(result: dict, cfg: dict, n_trades: int) -> None:
    bs   = result["_bs_array"]
    obs  = result["sharpe_obs"]
    lo   = result["ci_lower"]
    hi   = result["ci_upper"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(bs, bins=80, color="#4393c3", alpha=0.70, edgecolor="none")
    ax.axvline(obs, color="#e31a1c", lw=1.5, label=f"Observed Sharpe = {obs:.3f}")
    ax.axvline(lo,  color="#f59e0b", lw=1.2, ls="--", label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.axvline(hi,  color="#f59e0b", lw=1.2, ls="--")
    ax.axvline(0,   color="#888",    lw=0.7, ls=":")
    ax.axvline(1,   color="#888",    lw=0.7, ls=":", label="Sharpe = 1")

    tag_clean = cfg["tag"].replace("_", " ").title()
    ax.set_title(
        f"Bootstrap Sharpe — {tag_clean} {cfg['model_label']}\n"
        f"n_trades={n_trades}  P(>0)={result['p_positive']:.1%}  P(>1)={result['p_above_1']:.1%}",
        fontsize=8.5)
    ax.set_xlabel("Bootstrap Sharpe Ratio", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"sharpe_ci_{cfg['tag']}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    Plot saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Bootstrap Sharpe CI  (B={N_BOOTSTRAP:,})")
    print("=" * 60)

    rows = []
    for cfg in CONFIGS:
        print(f"\n  {cfg['tag']} / {cfg['model_label']}")
        strat_rets, signals = build_strategy_returns(cfg)
        if strat_rets is None:
            continue

        n_trades = int(np.sum(signals != 0))
        print(f"    n_periods={len(strat_rets)}, n_trades={n_trades}")

        result = bootstrap_sharpe(strat_rets)

        print(f"    Sharpe obs = {result['sharpe_obs']:.4f}")
        print(f"    95% CI     = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"    Median BS  = {result['median_bs']:.4f}")
        print(f"    P(>0)      = {result['p_positive']:.1%}")
        print(f"    P(>1)      = {result['p_above_1']:.1%}")

        plot_bootstrap(result, cfg, n_trades)

        rows.append({
            "tag":        cfg["tag"],
            "model":      cfg["model_label"],
            "n_trades":   n_trades,
            "sharpe_obs": result["sharpe_obs"],
            "ci_lower":   result["ci_lower"],
            "ci_upper":   result["ci_upper"],
            "median_bs":  result["median_bs"],
            "p_positive": result["p_positive"],
            "p_above_1":  result["p_above_1"],
            "n_boot":     result["n_valid_bs"],
        })

    if rows:
        out_csv = os.path.join(OUTPUT_DIR, "bootstrap_results.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  CSV saved: {out_csv}")
    else:
        print("\n  [WARN] Không có kết quả nào — kiểm tra predictions files.")

    print("\nDone.")


if __name__ == "__main__":
    main()
