"""
lag_analysis.py — Sprint 2: CCF Lag Validation
===============================================
Phân tích Cross-Correlation Function (CCF) giữa weather variables và return_future
để validate (hoặc bác bỏ) lag hardcode hiện tại trong data_integration.py.

Output:
    models/lag_analysis/lag_config_{crop}.json   — lag config để dùng trong data_integration
    models/lag_analysis/ccf_{crop}_{var}.png      — biểu đồ CCF (nếu matplotlib available)

Chạy (sau Sprint 1 — data expansion):
    python src/lag_analysis.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models", "12_lag_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_LAG_WEEKS  = 52       # scan lags 1..52 tuần
N_BOOTSTRAP    = 300      # số lần bootstrap để tính p-value
ROLLING_WINDOW = 52       # cửa sổ rolling CCF (tuần)
ROLLING_STEP   = 4        # bước dịch cửa sổ (tuần)
ALPHA          = 0.05     # ngưỡng p-value để gọi "significant"
STABILITY_FRAC = 0.6      # % cửa sổ rolling phải significant → gọi lag là "stable"

# Danh sách weather variables cần phân tích (prefix sau khi integrated)
WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm",
]


# ─── CCF ──────────────────────────────────────────────────────────────────────

def static_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = MAX_LAG_WEEKS,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """
    Tính Pearson correlation giữa x[t-lag] và y[t] cho lag = 1..max_lag.
    Bootstrap p-value: shuffle x n_bootstrap lần, lấy phân phối của |r| null.

    Returns: {lag: {"r": float, "p_value": float}}
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < max_lag + 20:
        return {}

    results = {}
    for lag in range(1, max_lag + 1):
        x_lag = x[: n - lag]
        y_lag = y[lag:]
        if len(x_lag) < 20:
            break

        r, _ = stats.pearsonr(x_lag, y_lag)

        # Bootstrap null distribution
        null_rs = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            x_shuf = rng.permutation(x_lag)
            r_null, _ = stats.pearsonr(x_shuf, y_lag)
            null_rs.append(abs(r_null))
        p_value = float(np.mean(np.array(null_rs) >= abs(r)))

        results[lag] = {"r": float(r), "p_value": p_value}

    return results


def rolling_ccf(
    x: pd.Series,
    y: pd.Series,
    lag: int,
    window: int = ROLLING_WINDOW,
    step: int = ROLLING_STEP,
) -> list[dict]:
    """
    CCF trong cửa sổ rolling để kiểm tra độ ổn định theo thời gian.
    Returns: list of {start_idx, r, p_value}
    """
    n = len(x)
    results = []
    for start in range(0, n - window - lag, step):
        end = start + window
        xi  = x.iloc[start : end - lag].values
        yi  = y.iloc[start + lag : end].values
        mask = ~(np.isnan(xi) | np.isnan(yi))
        if mask.sum() < 15:
            continue
        r, p = stats.pearsonr(xi[mask], yi[mask])
        results.append({"start_idx": start, "r": float(r), "p_value": float(p)})
    return results


# ─── Analysis per crop ────────────────────────────────────────────────────────

def find_weather_cols(df: pd.DataFrame) -> list[str]:
    """Tìm các cột weather trong DataFrame (có thể có prefix từ integration)."""
    found = []
    for base in WEATHER_VARS:
        matches = [c for c in df.columns if base in c and "lag" not in c]
        found.extend(matches)
    return list(dict.fromkeys(found))  # dedup, preserve order


def analyze_crop(crop: str, df: pd.DataFrame) -> dict:
    """
    Chạy static + rolling CCF cho mọi weather var vs return_future.
    Quyết định: shift_feature (stable lag) | rolling_corr (unstable) | drop (none).

    Returns: lag_config dict để save thành JSON.
    """
    if "return_future" not in df.columns:
        print(f"   [WARN] return_future không có trong {crop} data")
        return {}

    y = df["return_future"].values.astype(float)
    weather_cols = find_weather_cols(df)

    if not weather_cols:
        print(f"   [WARN] Không tìm thấy weather columns trong {crop}")
        return {}

    print(f"   Phân tích {len(weather_cols)} weather vars cho {crop}...")
    lag_config = {}

    for col in weather_cols:
        x = df[col].values.astype(float)
        ccf_results = static_ccf(x, y)
        if not ccf_results:
            continue

        # Tìm significant lags (p < ALPHA)
        sig_lags = [lag for lag, res in ccf_results.items() if res["p_value"] < ALPHA]
        if not sig_lags:
            continue

        # Với mỗi significant lag, kiểm tra rolling stability
        stable_lags   = []
        unstable_lags = []

        for lag in sig_lags:
            roll = rolling_ccf(df[col], df["return_future"], lag)
            if not roll:
                continue
            sig_windows = sum(1 for r in roll if r["p_value"] < ALPHA)
            frac = sig_windows / len(roll) if roll else 0.0
            if frac >= STABILITY_FRAC:
                stable_lags.append(lag)
            else:
                unstable_lags.append(lag)

        if not stable_lags and not unstable_lags:
            continue

        # Build action dict
        action = {}
        for lag in stable_lags:
            action[str(lag)] = "shift_feature"
        for lag in unstable_lags:
            action[str(lag)] = "rolling_corr"

        best_r = max(
            [ccf_results[lag]["r"] for lag in sig_lags],
            key=abs,
        )

        lag_config[col] = {
            "significant_lags_weeks": sig_lags,
            "stable_lags":            stable_lags,
            "unstable_lags":          unstable_lags,
            "best_abs_r":             round(abs(best_r), 4),
            "action":                 action,
        }

        print(f"      {col:<45}  sig_lags={sig_lags[:5]}  stable={stable_lags}  unstable={unstable_lags}")

    return lag_config


def try_plot_ccf(crop: str, col: str, ccf_results: dict) -> None:
    """Lưu biểu đồ CCF nếu matplotlib available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lags   = list(ccf_results.keys())
        rs     = [ccf_results[l]["r"] for l in lags]
        pvals  = [ccf_results[l]["p_value"] for l in lags]
        colors = ["red" if p < ALPHA else "steelblue" for p in pvals]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(lags, rs, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (weeks)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"CCF: {col} → return_future | {crop} weekly")
        ax.set_xticks(range(0, MAX_LAG_WEEKS + 1, 4))

        safe_col = col.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"ccf_{crop}_{safe_col}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
    except Exception:
        pass  # matplotlib not available or other error — skip silently


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    crops = ["coffee", "corn"]

    for crop in crops:
        # Dùng weekly data cho CCF (đủ lag range, ít noise hơn daily)
        path = os.path.join(INTEGRATED_DIR, f"integrated_{crop}_weekly.csv")
        if not os.path.exists(path):
            print(f"\n[SKIP] Không tìm thấy: {path}")
            print("       Chạy Sprint 1 (ingestion + preprocessing + integration) trước.")
            continue

        print(f"\n{'═'*64}")
        print(f"  CCF ANALYSIS — {crop.upper()} WEEKLY")
        print(f"{'═'*64}")

        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date").reset_index(drop=True)

        print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")

        lag_config = analyze_crop(crop, df)

        # Plot CCF charts
        for col in lag_config:
            x = df[col].values.astype(float)
            y = df["return_future"].values.astype(float)
            ccf_results = static_ccf(x, y)
            try_plot_ccf(crop, col, ccf_results)

        out_path = os.path.join(OUTPUT_DIR, f"lag_config_{crop}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lag_config, f, indent=2, ensure_ascii=False)

        n_vars    = len(lag_config)
        n_stable  = sum(len(v["stable_lags"]) for v in lag_config.values())
        n_rolling = sum(len(v["unstable_lags"]) for v in lag_config.values())
        print(f"\n   Kết quả: {n_vars} vars có significant lag")
        print(f"            {n_stable} stable lags (→ shift_feature)")
        print(f"            {n_rolling} unstable lags (→ rolling_corr)")
        print(f"   Saved: {os.path.basename(out_path)}")

    print(f"\n   Lag configs saved to: {OUTPUT_DIR}")
    print("   Tiếp theo: Sprint 3 — cập nhật data_integration.py dùng lag_config này.")


if __name__ == "__main__":
    main()
