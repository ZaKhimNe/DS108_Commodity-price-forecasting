"""
20b_ablation_calendar.py — Ablation: Synthetic vs LLM vs Hybrid Farming Calendar
==================================================================================
So sánh 3 cấu hình calendar features bằng LightGBM baseline:

  A — Synthetic only   : cal_* từ farming_preprocessing.py (rule-based)
  B — LLM only         : thay thế toàn bộ cal_* bằng LLM features
  C — Hybrid           : giữ cal_* + thêm LLM features

Output:
  models/ablation/calendar_A/  ← copy kết quả từ 14_lgbm_baseline/
  models/ablation/calendar_B/  ← LightGBM với LLM features
  models/ablation/calendar_C/  ← LightGBM với hybrid features
  models/ablation/calendar_comparison.json
  models/ablation/calendar_comparison.csv
"""

import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
LLM_DIR        = os.path.join(PROJECT_ROOT, "data", "raw", "farming", "llm_extracted")
ABLATION_DIR   = os.path.join(PROJECT_ROOT, "models", "ablation")
LGBM_DIR       = os.path.join(PROJECT_ROOT, "models", "14_lgbm_baseline")
TEMP_DIR       = os.path.join(PROJECT_ROOT, "data", "integrated_ablation_tmp")

os.makedirs(ABLATION_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

BASE_FILES = [
    "integrated_coffee_daily.csv",
    "integrated_coffee_weekly.csv",
    "integrated_corn_daily.csv",
    "integrated_corn_weekly.csv",
]

# ─── Load LLM Feature Tables ──────────────────────────────────────────────────

def load_llm_features() -> dict[str, pd.DataFrame]:
    corn_path   = os.path.join(LLM_DIR, "corn_weekly_llm.csv")
    coffee_path = os.path.join(LLM_DIR, "coffee_monthly_llm.csv")

    corn_llm = pd.read_csv(corn_path)
    corn_llm["Date"] = pd.to_datetime(corn_llm["Date"], utc=True)
    corn_llm = corn_llm.set_index("Date").add_prefix("llm_")

    coffee_llm = pd.read_csv(coffee_path)
    coffee_llm["Date"] = pd.to_datetime(coffee_llm["Date"], utc=True)
    coffee_llm = coffee_llm.set_index("Date").add_prefix("llm_")

    print(f"   Corn LLM features  : {list(corn_llm.columns)}")
    print(f"   Coffee LLM features: {list(coffee_llm.columns)}")
    return {"corn": corn_llm, "coffee": coffee_llm}


# ─── Build Modified Integrated CSV ────────────────────────────────────────────

def attach_llm_to_integrated(
    df: pd.DataFrame,
    llm_df: pd.DataFrame,
    mode: str,         # "llm_only" | "hybrid"
    is_weekly: bool,
) -> pd.DataFrame:
    """Attach LLM features to an integrated DataFrame.

    mode='llm_only'  → drop all cal_* columns first, then join LLM
    mode='hybrid'    → keep cal_* columns, join LLM on top
    """
    out = df.copy()
    date_col = None

    # Ensure Date index for join
    if "Date" in out.columns:
        date_col = out["Date"].copy()
        out["_date_idx"] = pd.to_datetime(out["Date"], utc=True)
        out = out.set_index("_date_idx")
        out.index.name = None

    # Drop cal_ columns for LLM-only
    if mode == "llm_only":
        cal_cols = [c for c in out.columns if c.startswith("cal_")]
        out = out.drop(columns=cal_cols)
        print(f"      Dropped {len(cal_cols)} cal_* columns")

    # Resample LLM features to weekly W-MON if needed
    if is_weekly:
        numeric_cols = llm_df.select_dtypes(include="number").columns
        llm_resampled = llm_df[numeric_cols].resample("W-MON").mean()
    else:
        llm_resampled = llm_df.copy()

    out = out.join(llm_resampled, how="left")
    out = out.ffill()

    # Restore Date column as first column
    if date_col is not None:
        out = out.drop(columns=["Date"], errors="ignore")
        out.insert(0, "Date", date_col.values)

    return out.reset_index(drop=True)


def build_variants(llm_tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Build LLM-only and hybrid CSVs. Returns {orig_fname: {B: path, C: path}}."""
    variant_paths: dict[str, dict[str, str]] = {}

    for fname in BASE_FILES:
        src = os.path.join(INTEGRATED_DIR, fname)
        if not os.path.exists(src):
            print(f"   [SKIP] {fname} not found")
            continue

        crop      = "coffee" if "coffee" in fname else "corn"
        is_weekly = "weekly" in fname
        df        = pd.read_csv(src)
        llm_df    = llm_tables[crop]

        variant_paths[fname] = {}

        for mode in ("llm_only", "hybrid"):
            tag     = "B" if mode == "llm_only" else "C"
            out_name = fname.replace(".csv", f"_{mode}.csv")
            out_path = os.path.join(TEMP_DIR, out_name)

            print(f"   Building {tag} ({mode}): {out_name}")
            modified = attach_llm_to_integrated(df, llm_df, mode, is_weekly)
            modified.to_csv(out_path, index=False)
            llm_col_count = len([c for c in modified.columns if c.startswith("llm_")])
            print(f"      Saved {len(modified)} rows × {len(modified.columns)} cols ({llm_col_count} LLM cols)")

            variant_paths[fname][tag] = out_path

    return variant_paths


# ─── Run LightGBM for a Single Experiment ─────────────────────────────────────

def run_lgbm_for_variant(
    input_path: str,
    output_dir: str,
) -> dict | None:
    """Import and run lgbm_baseline's run_pipeline on a custom input/output pair."""
    import importlib.util

    lgbm_src = os.path.join(SCRIPT_DIR, "modeling", "14_lgbm_baseline.py")
    spec     = importlib.util.spec_from_file_location("lgbm_mod", lgbm_src)
    mod      = importlib.util.module_from_spec(spec)

    # Patch module-level globals before exec
    os.makedirs(output_dir, exist_ok=True)
    spec.loader.exec_module(mod)

    # Override paths in the loaded module
    mod.INPUT_DIR  = os.path.dirname(input_path)
    mod.OUTPUT_DIR = output_dir

    fname = os.path.basename(input_path)
    try:
        res = mod.run_pipeline(fname)
        return res
    except Exception as exc:
        print(f"      [ERROR] {fname}: {exc}")
        return None


# ─── Collect Experiment A Baseline Results ────────────────────────────────────

def collect_experiment_a() -> dict[str, dict]:
    results = {}
    for fname in BASE_FILES:
        tag       = fname.replace(".csv", "")
        res_path  = os.path.join(LGBM_DIR, f"results_{tag}.json")
        if not os.path.exists(res_path):
            print(f"   [WARN] Experiment A results not found: {res_path}")
            continue
        with open(res_path) as f:
            raw = json.load(f)
        # Normalize: support both {"test": {...}} and {"splits": {"test": {...}}}
        if "splits" in raw:
            results[fname] = {k: v for k, v in raw["splits"].items()}
        else:
            results[fname] = raw
    return results


# ─── Build Comparison Table ───────────────────────────────────────────────────

def build_comparison_table(
    exp_a: dict[str, dict],
    exp_b: dict[str, dict],
    exp_c: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for fname in BASE_FILES:
        tag = fname.replace("integrated_", "").replace(".csv", "")
        for exp_label, exp_data in [("A_synthetic", exp_a), ("B_llm_only", exp_b), ("C_hybrid", exp_c)]:
            res = exp_data.get(fname)
            if res is None:
                rows.append({"dataset": tag, "experiment": exp_label,
                             "test_auc": None, "test_prauc": None, "test_f1": None})
                continue
            t = res.get("test", {})
            rows.append({
                "dataset":    tag,
                "experiment": exp_label,
                "test_auc":   round(t.get("auc_roc")  or 0.0, 4),
                "test_prauc": round(t.get("pr_auc")   or 0.0, 4),
                "test_f1":    round(t.get("f1")       or 0.0, 4),
            })

    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame) -> None:
    print(f"\n\n{'═'*72}")
    print("  ABLATION STUDY — Calendar Feature Comparison (LightGBM Baseline)")
    print(f"{'═'*72}")
    print(f"  {'Dataset':<30} {'Exp':<16} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1':>7}")
    print(f"  {'-'*69}")

    current_ds = None
    for _, row in df.iterrows():
        if row["dataset"] != current_ds:
            if current_ds is not None:
                print()
            current_ds = row["dataset"]
        auc   = row["test_auc"]   if row["test_auc"]   is not None else float("nan")
        prauc = row["test_prauc"] if row["test_prauc"] is not None else float("nan")
        f1    = row["test_f1"]    if row["test_f1"]    is not None else float("nan")
        print(
            f"  {row['dataset']:<30}"
            f" {row['experiment']:<16}"
            f" {auc:>8.4f}"
            f" {prauc:>8.4f}"
            f" {f1:>7.4f}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 64)
    print("  ABLATION CALENDAR — A vs B vs C")
    print("═" * 64)

    # Step 1: Collect Experiment A (existing results)
    print("\n[Step 1] Collecting Experiment A (synthetic only)...")
    exp_a = collect_experiment_a()
    print(f"   Found {len(exp_a)} result files")

    # Step 2: Load LLM features
    print("\n[Step 2] Loading LLM feature tables...")
    llm_tables = load_llm_features()

    # Step 3: Build B and C variant CSVs
    print("\n[Step 3] Building B (LLM only) and C (Hybrid) integrated CSVs...")
    variant_paths = build_variants(llm_tables)

    # Step 4: Run LightGBM for B and C
    exp_b: dict[str, dict] = {}
    exp_c: dict[str, dict] = {}

    for fname, paths in variant_paths.items():
        # Experiment B
        if "B" in paths:
            print(f"\n[Step 4B] Training B (LLM only) — {fname}")
            out_dir = os.path.join(ABLATION_DIR, "calendar_B")
            res = run_lgbm_for_variant(paths["B"], out_dir)
            if res:
                exp_b[fname] = res

        # Experiment C
        if "C" in paths:
            print(f"\n[Step 4C] Training C (Hybrid) — {fname}")
            out_dir = os.path.join(ABLATION_DIR, "calendar_C")
            res = run_lgbm_for_variant(paths["C"], out_dir)
            if res:
                exp_c[fname] = res

    # Step 5: Build comparison table
    print("\n[Step 5] Building comparison table...")
    comparison_df = build_comparison_table(exp_a, exp_b, exp_c)
    print_comparison(comparison_df)

    # Step 6: Save results
    out_json = os.path.join(ABLATION_DIR, "calendar_comparison.json")
    out_csv  = os.path.join(ABLATION_DIR, "calendar_comparison.csv")

    comparison_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "experiment_a": {k: v.get("test", {}) for k, v in exp_a.items()},
            "experiment_b": {k: v.get("test", {}) for k, v in exp_b.items()},
            "experiment_c": {k: v.get("test", {}) for k, v in exp_c.items()},
        }, f, indent=2)

    print(f"\n   Comparison saved:")
    print(f"   CSV  → {out_csv}")
    print(f"   JSON → {out_json}")

    # Clean up temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("\n[Done] Calendar ablation complete.")


if __name__ == "__main__":
    main()
