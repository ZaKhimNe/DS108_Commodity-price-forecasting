"""
Export static JSON files for DS108 dashboard.
Run from project root: python3 scripts/export_dashboard_data.py
Outputs 7 JSON files to ds108-dashboard/public/data/
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ds108-dashboard" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

DISPLAY = {
    "integrated_coffee_daily": "coffee_daily",
    "integrated_coffee_weekly": "coffee_weekly",
    "integrated_corn_daily": "corn_daily",
    "integrated_corn_weekly": "corn_weekly",
}


def save(name, data):
    path = OUT / name
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=str)
    print(f"  wrote {path.name} ({path.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────
# JSON 1: data_quality.json
# ─────────────────────────────────────────────────────────────
def export_data_quality():
    print("JSON 1: data_quality.json")
    rows = []
    for tag in TAGS:
        csv_path = ROOT / "data" / "integrated" / f"{tag}.csv"
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        n_test = n - n_train - n_val
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        col = "target_binary" if "target_binary" in df.columns else "return_future"
        rows.append(
            {
                "name": DISPLAY[tag],
                "tag": tag,
                "commodity": "coffee" if "coffee" in tag else "corn",
                "freq": "daily" if "daily" in tag else "weekly",
                "n_total": int(n),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "base_rate_train": round(float(train[col].mean()), 4) if col == "target_binary" else None,
                "base_rate_val": round(float(val[col].mean()), 4) if col == "target_binary" else None,
                "base_rate_test": round(float(test[col].mean()), 4) if col == "target_binary" else None,
            }
        )
    save("data_quality.json", {"datasets": rows})


# ─────────────────────────────────────────────────────────────
# JSON 2: feature_importance.json
# ─────────────────────────────────────────────────────────────
def classify_group(feature):
    f = feature.lower()
    if f.startswith("usd_") or f.startswith("vix_") or f.startswith("inf_"):
        return "macro"
    if f.startswith("cal_") or f in ("is_harvest", "is_planting"):
        return "farming"
    if any(
        f.startswith(p)
        for p in (
            "temperature_", "precipitation_", "et0_", "vpd_",
            "weekend_", "precip_", "temp_",
        )
    ):
        return "weather"
    return "market"


def export_feature_importance():
    print("JSON 2: feature_importance.json")
    result = {}
    for tag in TAGS:
        csv_path = ROOT / "models" / "14_lgbm_baseline" / f"importance_{tag}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "feature": str(row["feature"]),
                    "gain": round(float(row["importance_gain"]), 2),
                    "splits": int(row["importance_split"]),
                    "group": classify_group(str(row["feature"])),
                }
            )
        rows.sort(key=lambda x: x["gain"], reverse=True)
        result[DISPLAY[tag]] = rows
    save("feature_importance.json", result)


# ─────────────────────────────────────────────────────────────
# JSON 3: leakage_audit.json
# ─────────────────────────────────────────────────────────────
def export_leakage_audit():
    print("JSON 3: leakage_audit.json")
    modules = [
        {"module": "01 — Market Ingestion", "operation": "yfinance OHLCV download", "data_used": "Historical only", "type": "Nhân quả", "leakage": False, "note": "Giá đóng cửa lịch sử; không có look-ahead"},
        {"module": "02 — Weather Ingestion", "operation": "Open-Meteo archive download", "data_used": "Historical only", "type": "Nhân quả", "leakage": False, "note": "Dữ liệu thời tiết quá khứ"},
        {"module": "03 — Macro Ingestion", "operation": "Forex + CPI download", "data_used": "Historical + ffill", "type": "Nhân quả", "leakage": False, "note": "CPI shift +1m+12d trước ffill để simulate release lag"},
        {"module": "04 — Farming Ingestion", "operation": "Calendar flag generation", "data_used": "Static calendar", "type": "Nhân quả", "leakage": False, "note": "Flags cố định từ lịch nông vụ, không dùng giá"},
        {"module": "05 — MIQR (Anomaly)", "operation": "Rolling Q1/Q3 (w=15)", "data_used": "[t-14, t]", "type": "Nhân quả", "leakage": False, "note": "center=False; chỉ ngưỡng cục bộ"},
        {"module": "06 — Weather Preprocess", "operation": "Rolling impute (w=7)", "data_used": "[t-6, t]", "type": "Nhân quả", "leakage": False, "note": "Trailing window, không dùng giá tương lai"},
        {"module": "08 — Market ACU Filter", "operation": "RSI_14, SMA_20/50, BB, Vol", "data_used": "[t-W, t]", "type": "Nhân quả", "leakage": False, "note": "Tất cả rolling là trailing (center=False)"},
        {"module": "09 — Macro Preprocess", "operation": "RSI_14, Vol_20d USD", "data_used": "[t-W, t]", "type": "Nhân quả", "leakage": False, "note": "Trailing window"},
        {"module": "11 — Data Integration", "operation": "RSI_adj, BB_adj, EMA_adj", "data_used": "[t-W, t]", "type": "Nhân quả", "leakage": False, "note": "Computed trên currency_adjusted_close, trailing"},
        {"module": "13 — Tensor Packing", "operation": "MinMaxScaler fit", "data_used": "Train only", "type": "Nhân quả", "leakage": False, "note": "scaler.fit() chỉ trên train split; transform val/test"},
        {"module": "Target Variable", "operation": "return_future = (p_{t+7}/p_t - 1)", "data_used": "Giá t+7", "type": "Mục tiêu", "leakage": False, "note": "Đây là target hợp lệ; không rò rỉ vào features"},
    ]
    ablation = {
        "stacking_v7": {"coffee_daily": 0.709, "corn_daily": 0.662, "group": 1, "label": "Pipeline v7, threshold 2.5%"},
        "stacking_v1": {"coffee_daily": 0.564, "corn_daily": 0.443, "group": 1, "label": "Pipeline v1, threshold 5%"},
        "lgbm_baseline": {"coffee_daily": 0.405, "corn_daily": 0.475, "delta": 0.0, "group": 2, "label": "LGBM, correct pipeline"},
        "global_scaler": {"coffee_daily": 0.405, "corn_daily": 0.475, "delta": 0.0, "group": 3, "label": "Global scaler (fit toàn dataset)"},
        "no_embargo": {"coffee_daily": 0.405, "corn_daily": 0.475, "delta": 0.0, "group": 3, "label": "Không có embargo gap"},
        "center_rolling": {"coffee_daily": 0.653, "corn_daily": 0.703, "delta": 0.228, "group": 3, "label": "center=True (look-ahead rolling)"},
    }
    save("leakage_audit.json", {"modules": modules, "ablation": ablation})


# ─────────────────────────────────────────────────────────────
# JSON 4: model_results.json
# ─────────────────────────────────────────────────────────────
def _load_split(d, split):
    s = d.get("splits", {}).get(split, {})
    return {
        "auc": round(float(s.get("auc_roc", 0)), 4),
        "pr_auc": round(float(s.get("pr_auc", 0)), 4),
        "f1": round(float(s.get("f1", 0)), 4),
        "n": int(s.get("n_samples", 0)),
    }


def export_model_results():
    print("JSON 4: model_results.json")
    rows = []

    model_dirs = {
        "lgbm": "14_lgbm_baseline",
        "rf": "15_rf_baseline",
        "lstm": "16_lstm_hybrid",
        "tcn": "17_tcn_hybrid",
    }
    for tag in TAGS:
        for model_key, dir_name in model_dirs.items():
            path = ROOT / "models" / dir_name / f"results_{tag}.json"
            if not path.exists():
                continue
            d = json.load(open(path))
            train_s = _load_split(d, "train")
            val_s = _load_split(d, "val")
            test_s = _load_split(d, "test")
            rows.append(
                {
                    "tag": DISPLAY[tag],
                    "model": model_key,
                    "pipeline": "binary",
                    "test_auc": test_s["auc"],
                    "test_pr_auc": test_s["pr_auc"],
                    "test_f1": test_s["f1"],
                    "val_auc": val_s["auc"],
                    "train_auc": train_s["auc"],
                    "best_iter": int(d.get("best_iteration", d.get("best_window", 0))),
                    "n_test": test_s["n"],
                }
            )

    # Binary stacking
    for tag in TAGS:
        path = ROOT / "models" / "18_stacking_ensemble" / f"results_{tag}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        s = d.get("stacking", {})
        rows.append(
            {
                "tag": DISPLAY[tag],
                "model": "stack_binary",
                "pipeline": "binary",
                "test_auc": round(float(s.get("auc_roc", 0)), 4),
                "test_pr_auc": round(float(s.get("pr_auc", 0)), 4),
                "test_f1": round(float(s.get("f1", 0)), 4),
                "val_auc": None,
                "train_auc": None,
                "best_iter": None,
                "n_test": int(s.get("n", 0)),
            }
        )

    # Multiclass stacking
    for tag in TAGS:
        path = ROOT / "models" / "18b_stacking_multiclass" / f"results_{tag}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        s = d.get("stacking", {})
        rows.append(
            {
                "tag": DISPLAY[tag],
                "model": "stack_mc",
                "pipeline": "multiclass",
                "test_auc": round(float(s.get("auc_roc", 0)), 4),
                "test_pr_auc": None,
                "test_f1": round(float(s.get("f1", 0)), 4),
                "val_auc": None,
                "train_auc": None,
                "best_iter": None,
                "n_test": int(s.get("n", 0)),
            }
        )

    save("model_results.json", {"rows": rows})


# ─────────────────────────────────────────────────────────────
# JSON 5: backtest_results.json
# ─────────────────────────────────────────────────────────────
def _build_equity_curve(tag):
    pred_path = ROOT / "models" / "18b_stacking_multiclass" / f"test_predictions_{tag}.csv"
    int_path = ROOT / "data" / "integrated" / f"{tag}.csv"
    if not pred_path.exists() or not int_path.exists():
        return []
    preds = pd.read_csv(pred_path, parse_dates=["Date"])
    preds["Date"] = pd.to_datetime(preds["Date"], utc=True).dt.normalize()
    integ = pd.read_csv(int_path, parse_dates=["Date"])
    integ["Date"] = pd.to_datetime(integ["Date"], utc=True).dt.normalize()
    merged = preds.merge(integ[["Date", "return_future"]], on="Date", how="inner")
    merged = merged.dropna(subset=["return_future"])
    merged = merged.sort_values("Date").reset_index(drop=True)
    # Long on up (2), short on down (0), flat on (1)
    merged["position"] = merged["y_pred_stack"].map({2: 1, 0: -1, 1: 0})
    merged["daily_return"] = merged["position"] * merged["return_future"]
    merged["model_equity"] = (1 + merged["daily_return"]).cumprod()
    merged["bh_equity"] = (1 + merged["return_future"]).cumprod()
    return [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "model_equity": round(float(row["model_equity"]), 4),
            "bh_equity": round(float(row["bh_equity"]), 4),
        }
        for _, row in merged.iterrows()
    ]


def export_backtest_results():
    print("JSON 5: backtest_results.json")
    summary_path = ROOT / "models" / "19b_backtesting_mc" / "backtest_results_mc_all.csv"
    wf_path = ROOT / "models" / "21_walkforward_eval" / "walkforward_results.csv"

    summary = []
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            summary.append(
                {
                    "tag": DISPLAY.get(str(row["tag"]), str(row["tag"])),
                    "model": str(row["model"]),
                    "sharpe": round(float(row["sharpe"]), 4),
                    "mdd": round(float(row["max_drawdown"]), 4),
                    "win_rate": round(float(row["win_rate"]), 4),
                    "total_return": round(float(row["total_return"]), 4),
                    "bh_return": round(float(row["bh_total_return"]), 4),
                    "n_trades": int(row["n_trades"]),
                    "n_test": int(row["n_test"]),
                }
            )

    walkforward = []
    if wf_path.exists():
        df = pd.read_csv(wf_path)
        for _, row in df.iterrows():
            walkforward.append(
                {
                    "tag": DISPLAY.get(str(row["tag"]), str(row["tag"])),
                    "model": str(row["model"]),
                    "year": str(row["year"]),
                    "sharpe": round(float(row["sharpe"]), 4),
                    "mdd": round(float(row["mdd"]), 4),
                    "winrate": round(float(row["winrate"]), 4),
                    "n_trades": int(row["n_trades"]),
                }
            )

    print("  building equity curves...")
    equity_curves = {}
    for tag in TAGS:
        key = DISPLAY[tag]
        curve = _build_equity_curve(tag)
        if curve:
            equity_curves[key] = curve
            print(f"    {key}: {len(curve)} points")

    save("backtest_results.json", {"summary": summary, "walkforward": walkforward, "equity_curves": equity_curves})


# ─────────────────────────────────────────────────────────────
# JSON 6: hurdle_results.json
# ─────────────────────────────────────────────────────────────
def export_hurdle_results():
    print("JSON 6: hurdle_results.json")
    rows = []
    for tag in TAGS:
        path = ROOT / "models" / "22_hurdle_model" / f"results_{tag}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        rc = d.get("r_comparison", {})
        rows.append(
            {
                "tag": DISPLAY[tag],
                "single_r": round(float(rc.get("single_regression_14c", 0)), 4),
                "half_r": round(float(rc.get("half_hurdle_combined", 0)), 4),
                "full_r": round(float(rc.get("full_hurdle_combined", 0)), 4),
                "stage2a_r": round(float(rc.get("stage2_positive_only", 0)), 4),
                "n_test": int(d.get("n_pos_test", 0) + d.get("n_neg_test", 0)),
                "n_pos_test": int(d.get("n_pos_test", 0)),
                "n_neg_test": int(d.get("n_neg_test", 0)),
            }
        )
    save("hurdle_results.json", {"comparison": rows})


# ─────────────────────────────────────────────────────────────
# JSON 7: hurdle_histogram.json
# ─────────────────────────────────────────────────────────────
def export_hurdle_histogram():
    print("JSON 7: hurdle_histogram.json")
    result = {}
    for tag in TAGS:
        stats_path = ROOT / "models" / "22_hurdle_model" / f"zero_inflation_stats_{tag}.json"
        pred_path = ROOT / "models" / "22_hurdle_model" / f"test_predictions_{tag}.csv"
        if not stats_path.exists():
            continue
        stats = json.load(open(stats_path))
        key = DISPLAY[tag]
        result[key] = {}

        for split in ("full", "train", "val", "test"):
            s = stats.get(split, {})
            edges = s.get("hist_edges", [])
            counts = s.get("hist_counts", [])
            bin_centers = [round((edges[i] + edges[i + 1]) / 2, 4) for i in range(len(edges) - 1)]
            result[key][split] = {
                "n": int(s.get("n", 0)),
                "flat_pct": round(float(s.get("flat_pct", 0)), 4),
                "pos_pct": round(float(s.get("pos_pct", 0)), 4),
                "neg_pct": round(float(s.get("neg_pct", 0)), 4),
                "mean": round(float(s.get("mean", 0)), 5),
                "std": round(float(s.get("std", 0)), 5),
                "bin_centers": [round(float(c), 4) for c in bin_centers],
                "counts": [int(c) for c in counts],
            }

        # Scatter: test predictions — y_true vs full_hurdle_pred
        scatter = []
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            df = df.dropna(subset=["y_true", "full_hurdle_pred"])
            sample = df.sample(min(300, len(df)), random_state=42)
            threshold = 0.025
            for _, row in sample.iterrows():
                scatter.append(
                    {
                        "y_true": round(float(row["y_true"]), 5),
                        "hurdle_pred": round(float(row["full_hurdle_pred"]), 5),
                        "is_positive": bool(row["y_true"] > threshold),
                    }
                )
        result[key]["scatter"] = scatter

    save("hurdle_histogram.json", result)


if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    export_data_quality()
    export_feature_importance()
    export_leakage_audit()
    export_model_results()
    export_backtest_results()
    export_hurdle_results()
    export_hurdle_histogram()
    print("\nDone. All 7 JSON files written.")
