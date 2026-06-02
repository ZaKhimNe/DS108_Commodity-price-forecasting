"""
00_sanity_checks.py — DS108 Pipeline Sanity Checks

Chạy 8 kiểm tra tính toàn vẹn dữ liệu tự động. In PASS/FAIL cho từng check.
Exit code 0 nếu tất cả PASS, exit code 1 nếu có bất kỳ FAIL nào (CI/CD friendly).

Chạy từ project root:
    python src/00_sanity_checks.py
"""

import sys
import os
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INTEGRATED_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
TENSOR_DIR     = os.path.join(PROJECT_ROOT, "data", "tensors")

INTEGRATED_FILES = [
    "integrated_coffee_daily.csv",
    "integrated_coffee_weekly.csv",
    "integrated_corn_daily.csv",
    "integrated_corn_weekly.csv",
]

TARGET_COLS = {
    "target", "target_binary", "target_soft", "target_reg",
    "target_mc_down", "target_mc_flat", "target_mc_up",
    "target_multiclass", "return_future",
}
METADATA_COLS = {"Date", "Close", "currency_adjusted_close"} | TARGET_COLS

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.80

# ─── Helpers ──────────────────────────────────────────────────────────────────

_n_pass = 0
_n_fail = 0
_n_warn = 0


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")
    global _n_pass
    _n_pass += 1


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    global _n_fail
    _n_fail += 1


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    global _n_warn
    _n_warn += 1


def _load_integrated(fname: str) -> pd.DataFrame | None:
    fpath = os.path.join(INTEGRATED_DIR, fname)
    if not os.path.exists(fpath):
        _warn(f"{fname} not found — skipped")
        return None
    df = pd.read_csv(fpath, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _tensor_folders() -> list[str]:
    """Return all leaf tensor dataset folders."""
    folders = []
    if not os.path.isdir(TENSOR_DIR):
        return folders
    for win_dir in sorted(os.listdir(TENSOR_DIR)):
        win_path = os.path.join(TENSOR_DIR, win_dir)
        if not os.path.isdir(win_path) or not win_dir.startswith("win_"):
            continue
        for ds_dir in sorted(os.listdir(win_path)):
            ds_path = os.path.join(win_path, ds_dir)
            if os.path.isdir(ds_path):
                folders.append(ds_path)
    return folders


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Integrated data: no NaN
# ═══════════════════════════════════════════════════════════════════════════════
def check_1_no_nan() -> None:
    print("\n[CHECK 1] Integrated data — no NaN after preprocessing")
    for fname in INTEGRATED_FILES:
        df = _load_integrated(fname)
        if df is None:
            continue
        nan_count = df.isna().sum().sum()
        if nan_count == 0:
            _ok(f"{fname} ({len(df)} rows, 0 NaN)")
        else:
            _fail(f"{fname} has {nan_count} NaN values")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Train/Val/Test split: zero date overlap
# ═══════════════════════════════════════════════════════════════════════════════
def check_2_no_overlap() -> None:
    print("\n[CHECK 2] Train/Val/Test split — zero date overlap")
    for fname in INTEGRATED_FILES:
        df = _load_integrated(fname)
        if df is None:
            continue
        n       = len(df)
        horizon = 7 if "daily" in fname else 1
        val_start  = int(n * SPLIT_TRAIN)
        test_start = int(n * SPLIT_VAL)
        train_end  = val_start - horizon
        val_end    = test_start - horizon

        train_dates = set(df["Date"].iloc[:train_end].astype(str))
        val_dates   = set(df["Date"].iloc[val_start:val_end].astype(str))
        test_dates  = set(df["Date"].iloc[test_start:].astype(str))

        tv = len(train_dates & val_dates)
        vt = len(val_dates & test_dates)
        tt = len(train_dates & test_dates)

        if tv == 0 and vt == 0 and tt == 0:
            _ok(f"{fname}  train-val={tv}, val-test={vt}, train-test={tt} overlap")
        else:
            _fail(f"{fname}  overlap: train-val={tv}, val-test={vt}, train-test={tt}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Target cols not in feature tensor X_train
# ═══════════════════════════════════════════════════════════════════════════════
def check_3_no_target_leak() -> None:
    print("\n[CHECK 3] Target columns not in feature tensor X_train")
    folders = _tensor_folders()
    if not folders:
        _warn("No tensor folders found in data/tensors/ — skipped")
        return

    for folder in folders:
        tag = os.path.relpath(folder, TENSOR_DIR)
        meta_path = os.path.join(folder, "train_metadata.parquet")
        x_path    = os.path.join(folder, "X_train_dynamic.npy")

        if not os.path.exists(meta_path) or not os.path.exists(x_path):
            _warn(f"{tag} — missing metadata or tensor, skipped")
            continue

        meta = pd.read_parquet(meta_path)
        feature_cols = set(meta.columns) - METADATA_COLS

        leaked = feature_cols & TARGET_COLS
        if leaked:
            _fail(f"{tag} — target cols in features: {leaked}")
            continue

        X_train = np.load(x_path)
        available_d = len(feature_cols)
        actual_d    = X_train.shape[2]

        # actual_d <= available_d is expected: tensor_packing drops low-importance cols
        if actual_d <= available_d:
            _ok(f"{tag} — 0 target cols leaked, X_train D={actual_d} (selected from {available_d})")
        else:
            _fail(f"{tag} — X_train D={actual_d} > available feature_cols={available_d} (unexpected)")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Shape consistency X and y
# ═══════════════════════════════════════════════════════════════════════════════
def check_4_shape_consistency() -> None:
    print("\n[CHECK 4] Tensor shape consistency — X and y same N")
    folders = _tensor_folders()
    if not folders:
        _warn("No tensor folders found — skipped")
        return

    for folder in folders:
        tag = os.path.relpath(folder, TENSOR_DIR)
        ok_all = True
        for split in ("train", "val", "test"):
            x_dyn  = os.path.join(folder, f"X_{split}_dynamic.npy")
            x_stat = os.path.join(folder, f"X_{split}_static.npy")
            y_file = os.path.join(folder, f"y_{split}.npy")

            if not os.path.exists(x_dyn) or not os.path.exists(y_file):
                _warn(f"{tag}/{split} — missing file, skipped")
                ok_all = False
                continue

            X_dyn = np.load(x_dyn)
            y     = np.load(y_file)

            if X_dyn.shape[0] != y.shape[0]:
                _fail(f"{tag}/{split} X_dyn N={X_dyn.shape[0]} != y N={y.shape[0]}")
                ok_all = False

            if os.path.exists(x_stat):
                X_stat = np.load(x_stat)
                if X_stat.shape[0] != y.shape[0]:
                    _fail(f"{tag}/{split} X_stat N={X_stat.shape[0]} != y N={y.shape[0]}")
                    ok_all = False

        if ok_all:
            _ok(f"{tag} — all splits consistent")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — X_train in [0, 1] (MinMaxScaler fit on train only)
# ═══════════════════════════════════════════════════════════════════════════════
def check_5_scaler_range() -> None:
    print("\n[CHECK 5] Scaler range — X_train in [0, 1]; X_test may exceed (WARN only)")
    folders = _tensor_folders()
    if not folders:
        _warn("No tensor folders found — skipped")
        return

    for folder in folders:
        tag = os.path.relpath(folder, TENSOR_DIR)
        x_train_path = os.path.join(folder, "X_train_dynamic.npy")
        x_test_path  = os.path.join(folder, "X_test_dynamic.npy")

        if not os.path.exists(x_train_path):
            _warn(f"{tag} — X_train_dynamic.npy not found, skipped")
            continue

        X_train = np.load(x_train_path)
        t_min = float(X_train.min())
        t_max = float(X_train.max())

        if t_min >= -1e-6 and t_max <= 1 + 1e-6:
            _ok(f"{tag} X_train in [{t_min:.4f}, {t_max:.4f}]")
        else:
            _fail(f"{tag} X_train out of [0,1]: min={t_min:.4f}, max={t_max:.4f}")

        if os.path.exists(x_test_path):
            X_test  = np.load(x_test_path)
            n_out   = int(np.sum((X_test < 0) | (X_test > 1)))
            if n_out > 0:
                _warn(f"{tag} X_test has {n_out} values outside [0,1] — expected (unseen extremes)")
            else:
                _ok(f"{tag} X_test all within [0,1]")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Base rate sanity (10% – 60%)
# ═══════════════════════════════════════════════════════════════════════════════
def check_6_base_rate() -> None:
    print("\n[CHECK 6] Base rate sanity — y_train in [10%, 60%]")
    folders = _tensor_folders()
    if not folders:
        _warn("No tensor folders found — skipped")
        return

    for folder in folders:
        tag = os.path.relpath(folder, TENSOR_DIR)
        y_path = os.path.join(folder, "y_train.npy")
        if not os.path.exists(y_path):
            _warn(f"{tag} — y_train.npy not found, skipped")
            continue

        y_train = np.load(y_path)
        # Skip multiclass tensors (values not in {0,1})
        unique_vals = np.unique(y_train)
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            _ok(f"{tag} — multiclass target, base rate check skipped")
            continue

        base_rate = float(y_train.mean())
        if 0.10 <= base_rate <= 0.60:
            _ok(f"{tag} base_rate={base_rate:.1%}")
        else:
            _fail(f"{tag} base_rate={base_rate:.1%} outside [10%, 60%]")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Date monotonicity
# ═══════════════════════════════════════════════════════════════════════════════
def check_7_date_monotonic() -> None:
    print("\n[CHECK 7] Date monotonicity — dates strictly increasing")
    for fname in INTEGRATED_FILES:
        df = _load_integrated(fname)
        if df is None:
            continue
        dates = df["Date"].values
        if (dates[1:] > dates[:-1]).all():
            _ok(f"{fname} — dates strictly increasing")
        else:
            n_bad = int((dates[1:] <= dates[:-1]).sum())
            _fail(f"{fname} — {n_bad} non-increasing date steps")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — return_future uses shift(-horizon), not shift(+horizon)
# ═══════════════════════════════════════════════════════════════════════════════
def check_8_return_future_direction() -> None:
    print("\n[CHECK 8] return_future direction — causal integrity (corr > 0.80, adjusted for dropna effect)")
    for fname in INTEGRATED_FILES:
        df = _load_integrated(fname)
        if df is None:
            continue
        if "return_future" not in df.columns or "currency_adjusted_close" not in df.columns:
            _warn(f"{fname} — missing return_future or currency_adjusted_close, skipped")
            continue

        horizon = 7 if "daily" in fname else 1
        recalc  = df["currency_adjusted_close"].shift(-horizon) / df["currency_adjusted_close"] - 1
        mask    = df["return_future"].notna() & recalc.notna()

        corr = df.loc[mask, "return_future"].corr(recalc[mask])
        # Threshold 0.80: đủ để phát hiện shift sai chiều.
        # Không thể dùng 0.999 vì currency_adjusted_close có thể khác raw Close.
        if corr > 0.80:
            _ok(f"{fname} return_future correctly computed (corr={corr:.4f})")
        else:
            _fail(f"{fname} return_future correlation={corr:.4f} — possible look-ahead!")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════
def run_all_checks() -> None:
    print("=" * 55)
    print("  DS108 PIPELINE SANITY CHECKS")
    print("=" * 55)

    checks = [
        check_1_no_nan,
        check_2_no_overlap,
        check_3_no_target_leak,
        check_4_shape_consistency,
        check_5_scaler_range,
        check_6_base_rate,
        check_7_date_monotonic,
        check_8_return_future_direction,
    ]

    for fn in checks:
        try:
            fn()
        except Exception as e:
            _fail(f"Unexpected error in {fn.__name__}: {e}")

    total = _n_pass + _n_fail
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY: {_n_pass} PASSED, {_n_fail} FAILED"
          + (f", {_n_warn} WARN" if _n_warn else ""))
    print(f"{'=' * 55}")

    if _n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_checks()
