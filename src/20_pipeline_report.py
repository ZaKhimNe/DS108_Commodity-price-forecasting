"""
pipeline_report_v2.py — Sprint 6: Tự động sinh PIPELINE_REPORT_v2.md
=====================================================================
Đọc results JSON + val/test CSV từ tất cả models, tổng hợp thành báo cáo markdown.

Chạy sau khi tất cả models đã hoàn thành:
    python src/pipeline_report_v2.py
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_DIRS = {
    "lgbm":  os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline"),
    "lstm":  os.path.join(PROJECT_ROOT, "models", "16_lstm_hybrid"),
    "tcn":   os.path.join(PROJECT_ROOT, "models", "17_tcn_hybrid"),
    "rf":    os.path.join(PROJECT_ROOT, "models", "15_rf_baseline"),
    "stack": os.path.join(PROJECT_ROOT, "models", "18_stacking_ensemble"),
}
INTEGRATED_DIR  = os.path.join(PROJECT_ROOT, "data", "integrated")
BACKTEST_DIR    = os.path.join(PROJECT_ROOT, "models", "19_backtesting")
OUTPUT_PATH     = os.path.join(PROJECT_ROOT, "PIPELINE_REPORT_v2.md")

TAGS = [
    "integrated_coffee_daily",
    "integrated_coffee_weekly",
    "integrated_corn_daily",
    "integrated_corn_weekly",
]

DISPLAY_NAMES = {
    "integrated_coffee_daily":   "Coffee Daily",
    "integrated_coffee_weekly":  "Coffee Weekly",
    "integrated_corn_daily":     "Corn Daily",
    "integrated_corn_weekly":    "Corn Weekly",
}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_results(tag: str, model_key: str) -> dict:
    path = os.path.join(MODEL_DIRS[model_key], f"results_{tag}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_split_metrics(tag: str, model_key: str, split: str) -> dict:
    """Extract metrics for a specific split from results JSON."""
    data = load_results(tag, model_key)
    splits = data.get("splits", {})
    return splits.get(split, {})


def load_backtest(tag: str) -> pd.DataFrame | None:
    path = os.path.join(BACKTEST_DIR, f"backtest_results_{tag}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_integrated_stats(tag: str) -> dict:
    """Tính base rate và positive distribution từ integrated CSV."""
    path = os.path.join(INTEGRATED_DIR, f"{tag}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    n  = len(df)
    target_col = "target_binary" if "target_binary" in df.columns else "target"
    if target_col not in df.columns:
        return {"n_total": n}

    # Phân chia theo cùng logic 70/10/20
    horizon    = 7 if "daily" in tag else 1
    val_start  = int(n * 0.7)
    test_start = int(n * 0.8)
    train_df = df.iloc[: val_start  - horizon]
    val_df   = df.iloc[  val_start  : test_start - horizon]
    test_df  = df.iloc[  test_start :]

    def stats(split_df):
        if split_df.empty or target_col not in split_df.columns:
            return {"n": 0, "n_pos": 0, "base_rate": float("nan")}
        tgt = split_df[target_col].dropna()
        return {
            "n":         len(tgt),
            "n_pos":     int(tgt.sum()),
            "base_rate": round(float(tgt.mean()), 4),
        }

    return {
        "n_total":  n,
        "train":    stats(train_df),
        "val":      stats(val_df),
        "test":     stats(test_df),
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def _v(x, fmt=".4f") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return format(x, fmt)


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.1%}"


def _flag(v, warn_below=None, ok_above=None) -> str:
    """Thêm ⚠️ hoặc ✅ dựa trên ngưỡng."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " ❓"
    if warn_below is not None and v < warn_below:
        return " ⚠️"
    if ok_above is not None and v >= ok_above:
        return " ✅"
    return ""


# ─── Report sections ──────────────────────────────────────────────────────────

def section_header(lines: list) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    lines += [
        "# Báo Cáo Kết Quả Pipeline v2 — DS108",
        f"**Ngày chạy:** {today}  ",
        "**Dữ liệu:** Coffee (KC=F) & Corn (ZC=F) — Binary classification v2  ",
        "**Thay đổi so với v1:** LGBM params fix, OOF stacking, data expansion 2010-,",
        "corn weekly threshold 3%, 3 target formulations, RF base learner, backtesting",
        "",
        "---",
        "",
    ]


def section_data(lines: list) -> None:
    lines += [
        "## 1. Dữ Liệu & Phân Chia",
        "",
        "| Dataset | Total | Train | Val | Test | Base (Train) | Base (Val) | Base (Test) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        s = load_integrated_stats(tag)
        if not s:
            continue
        name  = DISPLAY_NAMES.get(tag, tag)
        tr    = s.get("train", {})
        vl    = s.get("val",   {})
        te    = s.get("test",  {})
        flag  = " ⚠️" if (te.get("n_pos", 1) == 0) else ""
        lines.append(
            f"| {name} | {s.get('n_total', '?')} | {tr.get('n','?')} | {vl.get('n','?')} "
            f"| {te.get('n','?')} | {_v(tr.get('base_rate'), '.1%')} "
            f"| {_v(vl.get('base_rate'), '.1%')} | **{_v(te.get('base_rate'), '.1%')}**{flag} |"
        )
    lines += ["", "---", ""]


def section_model_results(lines: list) -> None:
    lines += [
        "## 2. Kết Quả Mô Hình",
        "",
    ]
    model_keys = ["lgbm", "lstm", "tcn", "rf", "stack"]
    model_names = {"lgbm": "LightGBM", "lstm": "LSTM", "tcn": "TCN", "rf": "RF", "stack": "Stacking"}

    for tag in TAGS:
        name = DISPLAY_NAMES.get(tag, tag)
        lines += [f"### {name}", ""]
        lines += [
            "| Model | Train AUC | Val AUC | Test AUC | Test PR-AUC | Test F1 |",
            "|---|---|---|---|---|---|",
        ]
        for mk in model_keys:
            tr  = load_split_metrics(tag, mk, "train")
            vl  = load_split_metrics(tag, mk, "val")
            te  = load_split_metrics(tag, mk, "test")
            if not te:
                continue
            tr_auc = tr.get("auc_roc")
            vl_auc = vl.get("auc_roc")
            te_auc = te.get("auc_roc")
            gap    = (tr_auc - vl_auc) if (tr_auc and vl_auc) else None
            gap_flag = f" (gap={gap:.2f}⚠️)" if (gap and gap > 0.15) else ""
            lines.append(
                f"| {model_names[mk]} | {_v(tr_auc)}{gap_flag} | {_v(vl_auc)} "
                f"| **{_v(te_auc)}** | {_v(te.get('pr_auc'))} | {_v(te.get('f1'))} |"
            )
        lines += [""]

    lines += ["---", ""]


def section_backtesting(lines: list) -> None:
    lines += [
        "## 3. Backtesting",
        "",
    ]
    all_bt = []
    for tag in TAGS:
        df = load_backtest(tag)
        if df is not None and not df.empty:
            all_bt.append(df)

    if not all_bt:
        lines += ["*Chưa có dữ liệu backtesting — chạy backtesting_engine.py trước.*", "", "---", ""]
        return

    combined = pd.concat(all_bt, ignore_index=True)
    lines += [
        "| Dataset | Model | Sharpe | MDD | Win Rate | Return | BH Return |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in combined.iterrows():
        tag  = row.get("tag", "")
        name = DISPLAY_NAMES.get(tag, tag.replace("integrated_", ""))
        lines.append(
            f"| {name} | {row.get('model','?')} "
            f"| {_v(row.get('sharpe'), '.3f')} "
            f"| {_v(row.get('max_drawdown'), '.3f')} "
            f"| {_v(row.get('win_rate'), '.3f')} "
            f"| {_pct(row.get('total_return'))} "
            f"| {_pct(row.get('bh_total_return'))} |"
        )

    lines += ["", "---", ""]


def section_diagnosis(lines: list) -> None:
    lines += [
        "## 4. Chẩn đoán & Cảnh báo",
        "",
    ]
    issues = []
    for tag in TAGS:
        s = load_integrated_stats(tag)
        te = s.get("test", {})
        if te.get("n_pos", 1) == 0:
            issues.append(f"- **{DISPLAY_NAMES.get(tag, tag)}**: 0 positives trong test set → AUC = NaN")

        for mk in ["lgbm", "lstm", "tcn", "rf"]:
            tr = load_split_metrics(tag, mk, "train")
            vl = load_split_metrics(tag, mk, "val")
            tr_auc = tr.get("auc_roc")
            vl_auc = vl.get("auc_roc")
            if tr_auc and vl_auc and (tr_auc - vl_auc) > 0.15:
                issues.append(
                    f"- **{DISPLAY_NAMES.get(tag, tag)} / {mk.upper()}**: "
                    f"Overfit gap = {tr_auc - vl_auc:.3f} (train={tr_auc:.3f}, val={vl_auc:.3f})"
                )

    if not issues:
        lines += ["Không phát hiện vấn đề nghiêm trọng."]
    else:
        lines += issues

    lines += ["", "---", ""]


def section_conclusion(lines: list) -> None:
    lines += [
        "## 5. Kết Luận",
        "",
        "*(Cập nhật thủ công sau khi xem xét kết quả backtesting)*",
        "",
        "| Metric | Khuyến nghị |",
        "|--------|-------------|",
        "| Target mode tốt nhất | TBD — so sánh Sharpe: binary vs soft vs regression |",
        "| Model tốt nhất | TBD — xem Test AUC + Sharpe |",
        "| Corn Weekly | Threshold 3% → kiểm tra base rate val/test > 5% |",
        "",
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []

    section_header(lines)
    section_data(lines)
    section_model_results(lines)
    section_backtesting(lines)
    section_diagnosis(lines)
    section_conclusion(lines)

    report = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[V] Báo cáo đã lưu: {OUTPUT_PATH}")
    print(f"    Sections: header, data, model_results, backtesting, diagnosis, conclusion")


if __name__ == "__main__":
    main()
