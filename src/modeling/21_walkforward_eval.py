"""
21_walkforward_eval.py — Per-year Sharpe evaluation trên test period

Dùng predictions đã train (không retrain), slice theo calendar year để đánh giá
tính ổn định của signal qua các năm.

Production models:
  - coffee_daily  : MC Stack    (models/18b_stacking_multiclass/)
  - coffee_weekly : MC Stack    (models/18b_stacking_multiclass/)
  - corn_daily    : MC RF L/S   (models/15b_rf_multiclass/)

Gate:
  Sharpe > 0 trong ≥2/N năm → signal viable
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INTEGRATED_DIR   = os.path.join(ROOT, "data", "integrated")
STACK_DIR        = os.path.join(ROOT, "models", "18b_stacking_multiclass")
RF_DIR           = os.path.join(ROOT, "models", "15b_rf_multiclass")
OUT_DIR          = os.path.join(ROOT, "models", "21_walkforward_eval")
os.makedirs(OUT_DIR, exist_ok=True)

RISK_FREE_RATE = 0.02

# Threshold — matches 19b_backtesting_mc.py
PROB_UP_THR_DAILY    = 0.33
PROB_DOWN_THR_DAILY  = 0.33
PROB_UP_THR_WEEKLY   = 0.25
PROB_DOWN_THR_WEEKLY = 0.25

LONG_SHORT_TAGS = {"integrated_corn_daily", "integrated_corn_weekly"}


# ─── Metric helpers ───────────────────────────────────────────────────────────

def sharpe_ratio(strategy_returns, periods_per_year):
    r = np.asarray(strategy_returns, dtype=float)
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    excess = r - RISK_FREE_RATE / periods_per_year
    return float(excess.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(strategy_returns):
    equity = np.cumprod(1 + np.asarray(strategy_returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min()) if len(equity) > 0 else float("nan")


def win_rate(signals, returns):
    s = np.asarray(signals, dtype=float)
    r = np.asarray(returns, dtype=float)
    active = s != 0
    if active.sum() == 0:
        return float("nan")
    return float(((s[active] * r[active]) > 0).mean())


# ─── Signal generation ────────────────────────────────────────────────────────

def make_signals(prob_up, prob_down, tag, is_daily):
    thr_up   = PROB_UP_THR_DAILY   if is_daily else PROB_UP_THR_WEEKLY
    thr_down = PROB_DOWN_THR_DAILY if is_daily else PROB_DOWN_THR_WEEKLY
    if tag in LONG_SHORT_TAGS:
        return np.where(prob_up > thr_up, 1, np.where(prob_down > thr_down, -1, 0))
    else:
        return (prob_up > thr_up).astype(int)


# ─── Backtest one slice ───────────────────────────────────────────────────────

def backtest_slice(df_slice, tag, is_daily):
    """df_slice must have: prob_up, prob_down, return_future."""
    if len(df_slice) < 5:
        return None
    signals = make_signals(df_slice["prob_up"].values, df_slice["prob_down"].values, tag, is_daily)
    returns = df_slice["return_future"].values
    strat_r = signals.astype(float) * returns
    n_trades = int((signals != 0).sum())
    ppy = 52  # weekly periods per year (both daily after subsampling and weekly)
    return {
        "n"       : len(df_slice),
        "n_trades": n_trades,
        "sharpe"  : sharpe_ratio(strat_r, ppy) if n_trades > 0 else float("nan"),
        "mdd"     : max_drawdown(strat_r)       if n_trades > 0 else float("nan"),
        "winrate" : win_rate(signals, returns),
    }


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_price(tag):
    path = os.path.join(INTEGRATED_DIR, f"{tag}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df[["Date", "return_future"]].dropna()


def load_stack_preds(tag):
    path = os.path.join(STACK_DIR, f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    # stack columns
    if "y_prob_stack_up" not in df.columns:
        return None
    df = df.rename(columns={"y_prob_stack_up": "prob_up", "y_prob_stack_down": "prob_down"})
    return df[["Date", "prob_up", "prob_down"]]


def load_rf_preds(tag):
    path = os.path.join(RF_DIR, f"test_predictions_{tag}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(columns={"y_prob_rf_up": "prob_up", "y_prob_rf_down": "prob_down"})
    return df[["Date", "prob_up", "prob_down"]]


# ─── Main evaluation ──────────────────────────────────────────────────────────

MODELS = [
    # (tag, model_label, loader_fn)
    ("integrated_coffee_daily",  "MC Stack",  load_stack_preds),
    ("integrated_coffee_weekly", "MC Stack",  load_stack_preds),
    ("integrated_corn_daily",    "MC RF L/S", load_rf_preds),
]

rows = []

for tag, model_label, loader in MODELS:
    is_daily = "daily" in tag

    pred_df  = loader(tag)
    price_df = load_price(tag)

    if pred_df is None:
        print(f"[SKIP] {tag} — predictions not found")
        continue

    # Merge on Date
    merged = pred_df.merge(price_df, on="Date", how="inner")

    if len(merged) == 0:
        # Date column missing in old prediction files — tail-match
        n = len(pred_df)
        price_tail = price_df.tail(n).reset_index(drop=True)
        merged = pred_df.copy()
        merged["return_future"] = price_tail["return_future"].values
        # Attach approximate dates from price tail
        merged["Date"] = price_tail["Date"].values

    merged = merged.dropna(subset=["return_future"]).reset_index(drop=True)

    # Non-overlapping subsampling for daily (7-day forward return)
    if is_daily:
        merged = merged.iloc[::7].reset_index(drop=True)

    if "Date" not in merged.columns or merged["Date"].isna().all():
        print(f"[SKIP] {tag} — no Date column for year slicing")
        continue

    merged["year"] = pd.to_datetime(merged["Date"]).dt.year
    years = sorted(merged["year"].unique())

    # Full-period baseline
    full = backtest_slice(merged, tag, is_daily)
    if full:
        rows.append({"tag": tag, "model": model_label, "year": "ALL",
                     **full})

    for yr in years:
        sl = merged[merged["year"] == yr].reset_index(drop=True)
        res = backtest_slice(sl, tag, is_daily)
        if res:
            rows.append({"tag": tag, "model": model_label, "year": str(yr),
                         **res})

# ─── Report ───────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(OUT_DIR, "walkforward_results.csv"), index=False)

print()
print("═" * 80)
print("  WALKFORWARD EVALUATION — Per-Year Sharpe")
print("═" * 80)

for tag, grp in df_out.groupby("tag", sort=False):
    model = grp["model"].iloc[0]
    print(f"\n  {tag}  [{model}]")
    print(f"  {'Year':<8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Trades':>7} {'N':>5}")
    print("  " + "-" * 50)

    yearly = grp[grp["year"] != "ALL"].copy()
    all_row = grp[grp["year"] == "ALL"]

    pos_years = (yearly["sharpe"].fillna(-99) > 0).sum()
    total_years = len(yearly)

    for _, r in grp.iterrows():
        sharpe_str = f"{r['sharpe']:>8.3f}" if not np.isnan(r['sharpe']) else "     nan"
        mdd_str    = f"{r['mdd']:>7.1%}"   if not np.isnan(r['mdd'])    else "      -"
        wr_str     = f"{r['winrate']:>7.1%}" if not np.isnan(r['winrate']) else "      -"
        marker = " ✓" if (r["year"] != "ALL" and not np.isnan(r["sharpe"]) and r["sharpe"] > 0) else ""
        print(f"  {r['year']:<8} {sharpe_str}  {mdd_str}  {wr_str} {int(r['n_trades']):>7} {int(r['n']):>5}{marker}")

    # Gate decision
    if total_years > 0:
        pct = pos_years / total_years
        if pct >= 2/3:
            verdict = "VIABLE — signal consistent"
        elif pct >= 1/3:
            verdict = "MARGINAL — proceed with caution"
        else:
            verdict = "WEAK — may be bull-run artifact"
        print(f"\n  Gate: {pos_years}/{total_years} years Sharpe>0 → {verdict}")

print()
print(f"  Saved: {os.path.join(OUT_DIR, 'walkforward_results.csv')}")
print()
