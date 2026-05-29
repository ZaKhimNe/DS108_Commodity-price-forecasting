"""
generate_figures.py — DS108 IEEE paper figure generator

Produces 5 publication-quality PDFs in figures/:
  fig1_pipeline.pdf   — 8-stage pipeline architecture diagram
  fig2_ccf.pdf        — CCF lag analysis (coffee + corn weekly)
  fig3_split.pdf      — Train/Val/Test split with embargo gaps
  fig4_equity.pdf     — Equity curves (stack vs B&H, coffee daily + weekly)
  fig5_hurdle.pdf     — Hurdle model scatter: stage2a/2b/full/single (corn daily)

Run from project root:
  python generate_figures.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figures")
DATA_INT = os.path.join(ROOT, "data", "integrated")
MODELS   = os.path.join(ROOT, "models")

os.makedirs(FIGS, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.5,
    "lines.linewidth":  1.2,
})

# Greyscale-friendly palette (also works in color)
C = {
    "market":  "#2166ac",
    "weather": "#4dac26",
    "macro":   "#d6604d",
    "farming": "#762a83",
    "stack":   "#000000",
    "bh":      "#888888",
    "lstm":    "#1b7837",
    "tcn":     "#e08214",
    "lgbm":    "#4393c3",
    "rf":      "#984ea3",
    "hurdle":  "#d6604d",
    "single":  "#4393c3",
    "stage2a": "#1b7837",
    "stage2b": "#762a83",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ── Source boxes (left column) ──────────────────────────────────────────
    sources = [
        ("Market\n(yfinance)", C["market"],   5.0),
        ("Weather\n(Open-Meteo)", C["weather"], 3.8),
        ("Macro\n(BLS/yfinance)", C["macro"],  2.6),
        ("Farming\n(USDA/LLM)", C["farming"],  1.4),
    ]
    for label, color, y in sources:
        b = FancyBboxPatch((0.1, y - 0.45), 1.6, 0.9,
                           boxstyle="round,pad=0.05", linewidth=0.8,
                           edgecolor=color, facecolor=color + "22")
        ax.add_patch(b)
        ax.text(0.9, y, label, ha="center", va="center", fontsize=6.5,
                color=color, fontweight="bold")

    # ── Stage boxes (pipeline column) ──────────────────────────────────────
    stages = [
        (3.2, 5.2, "01–04\nIngestion",         "#555"),
        (3.2, 4.0, "05–10\nPreprocessing",     "#555"),
        (3.2, 2.8, "11–12\nIntegration + CCF", "#555"),
        (3.2, 1.6, "13\nTensor Packing",       "#555"),
    ]
    for x, y, label, color in stages:
        b = FancyBboxPatch((x - 0.7, y - 0.42), 1.4, 0.84,
                           boxstyle="round,pad=0.05", linewidth=0.8,
                           edgecolor=color, facecolor="#f0f0f0")
        ax.add_patch(b)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5,
                color=color)

    # arrows: sources → ingestion
    for _, color, sy in sources:
        ax.annotate("", xy=(2.45, 5.2), xytext=(1.7, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.7, mutation_scale=7))

    # arrows: ingestion → preprocessing → integration → tensor
    for (x1, y1, _, _), (x2, y2, _, _) in zip(stages, stages[1:]):
        ax.annotate("", xy=(x2, y2 + 0.42), xytext=(x1, y1 - 0.42),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=0.8, mutation_scale=8))

    # ── Model boxes (right column) ──────────────────────────────────────────
    models_stage6 = [
        (6.4, 5.1,  "LightGBM\n(Tầng 1)",    C["lgbm"]),
        (6.4, 4.0,  "Random Forest\n(Tầng 1)",C["rf"]),
        (6.4, 2.9,  "LSTM Hybrid\n(Tầng 2)",  C["lstm"]),
        (6.4, 1.8,  "TCN Hybrid\n(Tầng 2)",   C["tcn"]),
    ]
    for x, y, label, color in models_stage6:
        b = FancyBboxPatch((x - 0.75, y - 0.42), 1.5, 0.84,
                           boxstyle="round,pad=0.05", linewidth=0.8,
                           edgecolor=color, facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2,
                color=color)

    # tensor → each model
    for mx, my, _, color in models_stage6:
        ax.annotate("", xy=(mx - 0.76, my), xytext=(3.91, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.6, mutation_scale=7,
                                   connectionstyle="arc3,rad=0.0"))

    # ── Stacking box ─────────────────────────────────────────────────────────
    bx = FancyBboxPatch((8.4 - 0.75, 3.4 - 0.55), 1.5, 1.1,
                        boxstyle="round,pad=0.05", linewidth=1.0,
                        edgecolor=C["stack"], facecolor="#e8e8e8")
    ax.add_patch(bx)
    ax.text(8.4, 3.4, "Stacking\nEnsemble\n(Tầng 3)", ha="center", va="center",
            fontsize=6.5, color=C["stack"], fontweight="bold")

    for mx, my, _, color in models_stage6:
        ax.annotate("", xy=(7.64, 3.4), xytext=(mx + 0.75, my),
                    arrowprops=dict(arrowstyle="-|>", color="#888",
                                   lw=0.5, mutation_scale=6,
                                   connectionstyle="arc3,rad=0.0"))

    # ── Output ───────────────────────────────────────────────────────────────
    ax.text(8.4, 2.3, "AUC-ROC / PR-AUC\nBacktest + Hurdle", ha="center",
            va="top", fontsize=6.2, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff8e1", ec="#f59e0b", lw=0.8))
    ax.annotate("", xy=(8.4, 2.6), xytext=(8.4, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#f59e0b",
                                lw=0.8, mutation_scale=8))

    # ── Title + figure label ─────────────────────────────────────────────────
    ax.set_title("DS108 Multi-Source Commodity Forecasting Pipeline",
                 fontsize=9, pad=6, fontweight="bold")
    ax.text(0.01, 0.01, "Fig. 1.", transform=ax.transAxes,
            fontsize=7, style="italic", va="bottom")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_pipeline.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — CCF Lag Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _ccf_bootstrap(x, y, max_lag, n_boot=300, alpha=0.05):
    """Return arrays: lags, r_obs, significant (bool), ci_bound."""
    lags, r_obs, p_vals = [], [], []
    rng = np.random.default_rng(42)
    n = len(x)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        xlag = x[:-lag]
        yfut = y[lag:]
        mask = ~(np.isnan(xlag) | np.isnan(yfut))
        if mask.sum() < 20:
            continue
        r, _ = stats.pearsonr(xlag[mask], yfut[mask])
        # bootstrap under H0: shuffle x
        r_null = np.array([
            stats.pearsonr(rng.permutation(xlag[mask]), yfut[mask])[0]
            for _ in range(n_boot)
        ])
        p = np.mean(np.abs(r_null) >= np.abs(r))
        lags.append(lag)
        r_obs.append(r)
        p_vals.append(p)
    lags    = np.array(lags)
    r_obs   = np.array(r_obs)
    p_vals  = np.array(p_vals)
    sig     = p_vals < alpha
    # 95th percentile of null distribution as CI band
    ci_band = np.quantile(np.abs(r_null), 1 - alpha) if len(r_null) else 0.1
    return lags, r_obs, sig, ci_band


def fig2_ccf():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), sharex=False)

    # bio_lag: domain-knowledge biological lag to mark as reference (None = skip)
    configs = [
        ("integrated_coffee_weekly.csv", "temperature_2m_max",
         "Coffee — Temperature (Max)", C["weather"], 52, 34),
        ("integrated_coffee_weekly.csv", "et0_fao_evapotranspiration",
         "Coffee — ET0 (Evapotranspiration)", C["macro"], 52, 34),
        ("integrated_corn_weekly.csv",   "precipitation_sum",
         "Corn — Precipitation", C["market"], 20, 9),
        ("integrated_corn_weekly.csv",   "temperature_2m_max",
         "Corn — Temperature (Max)", C["farming"], 20, 9),
    ]

    for ax, (fname, xcol, title, color, max_lag, bio_lag) in zip(axes.flat, configs):
        fpath = os.path.join(DATA_INT, fname)
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f"Data not found:\n{fname}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="red")
            ax.set_title(title)
            continue

        df = pd.read_csv(fpath, parse_dates=["Date"])
        if xcol not in df.columns or "return_future" not in df.columns:
            ax.text(0.5, 0.5, f"Column missing:\n{xcol}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)
            ax.set_title(title)
            continue

        x = df[xcol].values.astype(float)
        y = df["return_future"].values.astype(float)

        lags, r_obs, sig, ci_band = _ccf_bootstrap(x, y, max_lag, n_boot=200)

        # stem plot
        markerline, stemlines, baseline = ax.stem(lags, r_obs, linefmt="-",
                                                   markerfmt="o", basefmt="k-")
        plt.setp(stemlines, color=color, linewidth=0.7, alpha=0.55)
        plt.setp(markerline, color=color, markersize=2.5)
        plt.setp(baseline, linewidth=0.6, color="#888")

        # significant lags — filled, larger marker
        sig_idx = np.where(sig)[0]
        if len(sig_idx):
            ax.scatter(lags[sig_idx], r_obs[sig_idx],
                       color=color, s=20, zorder=5,
                       label=f"p<0.05 (n={len(sig_idx)})")

        # Bootstrap CI band
        ax.axhline( ci_band, color="#aaa", ls="--", lw=0.7,
                   label=f"95% CI (+/-{ci_band:.3f})")
        ax.axhline(-ci_band, color="#aaa", ls="--", lw=0.7)
        ax.axhline(0, color="#888", lw=0.5)

        # Domain-knowledge biological lag reference line
        if bio_lag and bio_lag <= max_lag:
            ax.axvline(bio_lag, color="#d62728", ls=":", lw=1.0, alpha=0.8,
                       label=f"Bio. lag {bio_lag}w (domain)")
            ax.text(bio_lag + 0.5, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else -0.01,
                    f"{bio_lag}w", fontsize=5.5, color="#d62728", va="top")

        # Annotate peak significant lag (or overall peak if none significant)
        if len(r_obs):
            pk_pool = sig_idx if len(sig_idx) else np.arange(len(r_obs))
            pk = pk_pool[np.argmax(np.abs(r_obs[pk_pool]))]
            ax.annotate(f"lag {lags[pk]}\nr={r_obs[pk]:.3f}",
                        xy=(lags[pk], r_obs[pk]),
                        xytext=(lags[pk] + max(2, max_lag // 10),
                                r_obs[pk] + np.sign(r_obs[pk]) * 0.035),
                        fontsize=5.5, color=color,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6,
                                        mutation_scale=6))

        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("Lag (weeks)", fontsize=7)
        ax.set_ylabel("Pearson r", fontsize=7)
        ax.legend(fontsize=5.5, framealpha=0.4, loc="upper right")

    fig.suptitle(
        "Fig. 2 — CCF: Weather Variables vs. Future Return (dashed red = biological lag)",
        fontsize=8.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_ccf.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Train / Val / Test Split with Embargo Gaps
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_split():
    datasets = [
        ("integrated_coffee_daily.csv",  "Coffee Daily"),
        ("integrated_coffee_weekly.csv", "Coffee Weekly"),
        ("integrated_corn_daily.csv",    "Corn Daily"),
        ("integrated_corn_weekly.csv",   "Corn Weekly"),
    ]

    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.0, 4.5))

    for ax, (fname, title) in zip(axes, datasets):
        fpath = os.path.join(DATA_INT, fname)
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f"Not found: {fname}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_yticks([])
            continue

        df = pd.read_csv(fpath, parse_dates=["Date"])
        df = df.dropna(subset=["return_future"])
        n = len(df)
        horizon = 7 if "daily" in fname else 1

        val_start  = int(n * 0.70)
        test_start = int(n * 0.80)

        train_end  = val_start - horizon
        val_end    = test_start - horizon

        dates = df["Date"].values
        d0  = pd.Timestamp(dates[0])
        d_te = pd.Timestamp(dates[train_end])
        d_vs = pd.Timestamp(dates[val_start])
        d_ve = pd.Timestamp(dates[val_end])
        d_ts = pd.Timestamp(dates[test_start])
        d_tn = pd.Timestamp(dates[-1])

        total_days = (d_tn - d0).days or 1

        def frac(d):
            return (d - d0).days / total_days

        segments = [
            (d0,    d_te,  "#4393c3", "Train (70%)", 0.55),
            (d_te,  d_vs,  "#d0d0d0", f"Embargo ({horizon})", 0.5),
            (d_vs,  d_ve,  "#f59e0b", "Val (10%)", 0.55),
            (d_ve,  d_ts,  "#d0d0d0", f"Embargo ({horizon})", 0.5),
            (d_ts,  d_tn,  "#e31a1c", "Test (20%)", 0.55),
        ]

        for (da, db, color, label, alpha) in segments:
            x0, x1 = frac(da), frac(db)
            ax.barh(0, x1 - x0, left=x0, height=0.6,
                    color=color, alpha=alpha, edgecolor="none")
            width = x1 - x0
            if width > 0.04 and label != f"Embargo ({horizon})":
                ax.text(x0 + width / 2, 0, label, ha="center", va="center",
                        fontsize=6.5, color="white" if color != "#f59e0b" else "#333",
                        fontweight="bold")

        # Date tick labels at boundaries
        for d, lbl in [(d0, d0.strftime("%Y")),
                       (d_vs, d_vs.strftime("%Y-%m")),
                       (d_ts, d_ts.strftime("%Y-%m")),
                       (d_tn, d_tn.strftime("%Y-%m"))]:
            ax.axvline(frac(d), color="#666", lw=0.5, ls=":")
            ax.text(frac(d), -0.42, lbl, ha="center", va="top",
                    fontsize=5.5, color="#444")

        ax.set_title(f"{title}  (N={n})", fontsize=7.5, pad=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4393c3", alpha=0.55, label="Train (70%)"),
        mpatches.Patch(color="#f59e0b", alpha=0.55, label="Validation (10%)"),
        mpatches.Patch(color="#e31a1c", alpha=0.55, label="Test (20%)"),
        mpatches.Patch(color="#d0d0d0", alpha=0.5,  label="Embargo gap"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig. 3 — Temporal Train/Val/Test Split with Embargo Gaps",
                 fontsize=9, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIGS, "fig3_split.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Equity Curves
# ═══════════════════════════════════════════════════════════════════════════════
def _build_equity(pred_csv, integrated_csv, model_col="y_pred_stack",
                  prob_col="y_prob_stack", threshold=0.47):
    """Reconstruct equity curve from stacking test predictions + return_future."""
    preds = pd.read_csv(pred_csv, parse_dates=["Date"])
    integ = pd.read_csv(integrated_csv, parse_dates=["Date"])

    # normalize timezone
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.tz_localize(None)
    integ["Date"] = pd.to_datetime(integ["Date"]).dt.tz_localize(None)

    merged = preds.merge(integ[["Date", "return_future"]], on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # signal: trade when prob > threshold
    merged["signal"] = (merged[prob_col] >= threshold).astype(int)
    merged["strat_ret"] = merged["signal"] * merged["return_future"]
    merged["bh_ret"]    = merged["return_future"]

    merged["equity_strat"] = (1 + merged["strat_ret"]).cumprod()
    merged["equity_bh"]    = (1 + merged["bh_ret"]).cumprod()
    return merged


def _build_equity_ls(pred_csv, integrated_csv, step=7):
    """Build L/S equity for MC RF: long=Up(2), short=Down(0), flat=Flat(1).
    step: non-overlapping stride (7 for daily to match backtesting engine)."""
    preds = pd.read_csv(pred_csv, parse_dates=["Date"])
    integ = pd.read_csv(integrated_csv, parse_dates=["Date"])
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.tz_localize(None)
    integ["Date"] = pd.to_datetime(integ["Date"]).dt.tz_localize(None)
    merged = preds.merge(integ[["Date", "return_future"]], on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    # Non-overlapping sampling matches backtesting engine (iloc[::step])
    merged = merged.iloc[::step].reset_index(drop=True)
    # signal: +1 long (Up), -1 short (Down), 0 flat
    merged["signal"] = merged["y_pred_rf"].map({2: 1, 0: -1, 1: 0})
    merged["strat_ret"] = merged["signal"] * merged["return_future"]
    merged["bh_ret"]    = merged["return_future"]
    merged["equity_strat"] = (1 + merged["strat_ret"]).cumprod()
    merged["equity_bh"]    = (1 + merged["bh_ret"]).cumprod()
    return merged


def fig4_equity():
    """Two-panel equity figure matching paper caption:
    (a) Coffee Weekly Binary RF  — alpha +16.4 pp
    (b) Corn Daily MC RF L/S     — alpha +57.4 pp
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ── Panel A: Coffee Weekly RF (binary) ──────────────────────────────────
    ax = axes[0]
    pred_a = os.path.join(MODELS, "15_rf_baseline",
                          "test_predictions_integrated_coffee_weekly.csv")
    int_a  = os.path.join(DATA_INT, "integrated_coffee_weekly.csv")

    if os.path.exists(pred_a) and os.path.exists(int_a):
        try:
            df = _build_equity(pred_a, int_a,
                               model_col="y_pred_rf",
                               prob_col="y_prob_rf",
                               threshold=0.37)
            ax.plot(df["Date"], df["equity_strat"],
                    color=C["rf"], lw=1.3, label="RF Binary")
            ax.plot(df["Date"], df["equity_bh"],
                    color=C["bh"], lw=1.0, ls="--", label="Buy & Hold")
            ax.axhline(1.0, color="#ccc", lw=0.5)
            final_s = df["equity_strat"].iloc[-1]
            final_b = df["equity_bh"].iloc[-1]
            alpha_pp = (final_s - final_b) * 100
            ax.set_title(
                f"(a) Coffee Weekly — Binary RF\n"
                f"RF +{(final_s-1)*100:.1f}%  B&H +{(final_b-1)*100:.1f}%  "
                f"(alpha {alpha_pp:+.1f} pp)",
                fontsize=7.5)
            ax.legend(loc="upper left", fontsize=6)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)
    else:
        ax.text(0.5, 0.5, "Data not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)

    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Cumulative Return", fontsize=7)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)

    # ── Panel B: Corn Daily MC RF L/S ────────────────────────────────────────
    ax = axes[1]
    pred_b = os.path.join(MODELS, "15b_rf_multiclass",
                          "test_predictions_integrated_corn_daily.csv")
    int_b  = os.path.join(DATA_INT, "integrated_corn_daily.csv")

    if os.path.exists(pred_b) and os.path.exists(int_b):
        try:
            df = _build_equity_ls(pred_b, int_b)
            ax.plot(df["Date"], df["equity_strat"],
                    color=C["rf"], lw=1.3, label="MC RF L/S")
            ax.plot(df["Date"], df["equity_bh"],
                    color=C["bh"], lw=1.0, ls="--", label="Buy & Hold")
            ax.axhline(1.0, color="#ccc", lw=0.5)
            final_s = df["equity_strat"].iloc[-1]
            final_b = df["equity_bh"].iloc[-1]
            alpha_pp = (final_s - final_b) * 100
            ax.set_title(
                f"(b) Corn Daily — MC RF L/S\n"
                f"RF {(final_s-1)*100:+.1f}%  B&H {(final_b-1)*100:+.1f}%  "
                f"(alpha {alpha_pp:+.1f} pp)",
                fontsize=7.5)
            ax.legend(loc="upper left", fontsize=6)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)
    else:
        ax.text(0.5, 0.5, "Data not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)

    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Cumulative Return", fontsize=7)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)

    fig.suptitle("Fig. 4 — Equity Curves on Test Set (Production Candidates)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_equity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Hurdle Model Scatter (Corn Daily)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_hurdle():
    hurdle_csvs = [
        (os.path.join(MODELS, "22_hurdle_model",
                      "test_predictions_integrated_corn_daily.csv"), "Corn Daily"),
        (os.path.join(MODELS, "22_hurdle_model",
                      "test_predictions_integrated_coffee_daily.csv"), "Coffee Daily"),
    ]
    # 2x3 layout: Stage2a | Stage2b | Full Hurdle
    # Stage 2a has best_iteration=1 (constant predictor, 5-7 unique values).
    # Displayed as a strip+box plot instead of scatter to honestly show
    # the near-constant prediction range and actual return distribution.
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.5))

    def scatter_panel(ax, x, y, color, label, xlabel, ylabel):
        mask = ~(np.isnan(x) | np.isnan(y))
        xm, ym = x[mask], y[mask]
        if len(xm) < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            return
        r, p = stats.pearsonr(xm, ym)
        ax.scatter(xm, ym, alpha=0.22, s=7, color=color, rasterized=True)
        m, b = np.polyfit(xm, ym, 1)
        xl = np.linspace(xm.min(), xm.max(), 100)
        ax.plot(xl, m * xl + b, color=color, lw=1.2)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvline(0, color="#ccc", lw=0.5)
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else " n.s."))
        ax.set_title(f"{label}\nr={r:+.3f}{stars}  n={mask.sum()}",
                     fontsize=7.5, pad=3)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)

    def stage2a_panel(ax, x_pred, y_actual, color, label):
        """Stage 2a: near-constant predictor → show as box-per-bin with jitter."""
        mask = ~(np.isnan(x_pred) | np.isnan(y_actual))
        xm = x_pred[mask]
        ym = y_actual[mask]
        if len(xm) < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            return
        r, p = stats.pearsonr(xm, ym)

        # Bin by predicted value (few unique values → each unique = one bin)
        unique_x = np.sort(np.unique(np.round(xm, 4)))
        positions = np.arange(len(unique_x))
        groups = [ym[np.round(xm, 4) == ux] for ux in unique_x]

        # box plot per bin
        bp = ax.boxplot(groups, positions=positions, widths=0.4,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color + "55")
            patch.set_edgecolor(color)
            patch.set_linewidth(0.8)
        for item in ["whiskers", "caps"]:
            for line in bp[item]:
                line.set_color(color)
                line.set_linewidth(0.7)

        # jitter overlay
        rng = np.random.default_rng(42)
        for i, (pos, grp) in enumerate(zip(positions, groups)):
            jitter = rng.uniform(-0.18, 0.18, len(grp))
            ax.scatter(pos + jitter, grp, alpha=0.3, s=5, color=color,
                       rasterized=True, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{v:.3f}" for v in unique_x],
                           fontsize=5, rotation=40)
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else " n.s."))
        n_unique = len(unique_x)
        ax.set_title(f"{label}\nr={r:+.3f}{stars}  n={mask.sum()}\n"
                     f"({n_unique} unique pred. values — near-constant)",
                     fontsize=6.8, pad=3)
        ax.set_xlabel("Predicted return (binned)", fontsize=7)
        ax.set_ylabel("Actual return", fontsize=7)

    for row_idx, (hcsv, crop_label) in enumerate(hurdle_csvs):
        ax_2a = axes[row_idx][0]
        ax_2b = axes[row_idx][1]
        ax_fh = axes[row_idx][2]

        if not os.path.exists(hcsv):
            for ax in [ax_2a, ax_2b, ax_fh]:
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            continue

        df = pd.read_csv(hcsv, parse_dates=["Date"])
        cols_needed = ["y_true", "stage2_magnitude", "stage2_neg_magnitude",
                       "full_hurdle_pred", "target_binary"]
        if any(c not in df.columns for c in cols_needed):
            for ax in [ax_2a, ax_2b, ax_fh]:
                ax.text(0.5, 0.5, "Missing cols", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            continue

        pos_mask = df["target_binary"].values == 1.0
        neg_mask = df["target_binary"].values == 0.0

        # Panel A: Stage 2a — box+jitter per unique predicted value
        stage2a_panel(ax_2a,
                      df.loc[pos_mask, "stage2_magnitude"].values,
                      df.loc[pos_mask, "y_true"].values,
                      C["stage2a"],
                      f"{crop_label} — Stage 2a (Up)")

        # Panel B: Stage 2b — scatter, abs(y_true) → r matches JSON (+0.371)
        scatter_panel(ax_2b,
                      df.loc[neg_mask, "stage2_neg_magnitude"].values,
                      np.abs(df.loc[neg_mask, "y_true"].values),
                      C["stage2b"],
                      f"{crop_label} — Stage 2b (Down)",
                      xlabel="Predicted |return|", ylabel="Actual |return|")

        # Panel C: Full hurdle combined
        scatter_panel(ax_fh,
                      df["full_hurdle_pred"].values,
                      df["y_true"].values,
                      C["hurdle"],
                      f"{crop_label} — Full Hurdle",
                      xlabel="Predicted return", ylabel="Actual return")

    fig.text(0.01, 0.73, "Corn Daily",   va="center", rotation=90,
             fontsize=8, fontweight="bold", color="#333")
    fig.text(0.01, 0.25, "Coffee Daily", va="center", rotation=90,
             fontsize=8, fontweight="bold", color="#333")

    fig.suptitle("Fig. 5 — Hurdle Model: Stage 2a / Stage 2b / Full Hurdle",
                 fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    out = os.path.join(FIGS, "fig5_hurdle.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures -> figures/")
    fig1_pipeline()
    fig2_ccf()
    fig3_split()
    fig4_equity()
    fig5_hurdle()
    print("\nDone. All PDFs written to figures/")
