"""
17c_tcn_regression.py — Tầng 2c: TCN Regression
=================================================
Giống 16c_lstm_regression.py nhưng dùng TCN architecture.
HuberLoss, no sigmoid, best window by val MAE.

Chạy:
    python src/modeling/17c_tcn_regression.py
"""

import os
import json
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors_reg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "17c_tcn_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = dict(
    n_filters     = 64,
    kernel_size   = 3,
    n_blocks      = 4,
    head_hidden   = 64,
    dropout       = 0.20,

    lr            = 5e-4,
    weight_decay  = 1e-4,
    max_epochs    = 120,
    batch_daily   = 64,
    batch_weekly  = 32,
    patience      = 15,
    lr_patience   = 5,
    clip_norm     = 1.0,
    huber_delta   = 0.05,

    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)

FILES_DAILY  = ["integrated_coffee_daily",  "integrated_corn_daily"]
FILES_WEEKLY = ["integrated_coffee_weekly", "integrated_corn_weekly"]

SIGNAL_THR = 0.05


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, X_dyn, X_stat, y):
        # TCN: channel-first (N, D, W)
        self.X_dyn  = torch.FloatTensor(X_dyn).permute(0, 2, 1)
        self.X_stat = torch.FloatTensor(X_stat) if X_stat.ndim == 2 and X_stat.shape[1] > 0 \
                      else torch.zeros(len(X_dyn), 0)
        self.y      = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_dyn[idx], self.X_stat[idx], self.y[idx]


def make_loader(X_dyn, X_stat, y, batch_size, shuffle=False):
    return DataLoader(HybridDataset(X_dyn, X_stat, y),
                      batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ─── TCN Building Blocks ──────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)
        self.residual_proj = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                              if in_channels != out_channels else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop2(out)
        return self.relu(out + residual)


# ─── TCN Hybrid Model ─────────────────────────────────────────────────────────

class TCNHybridReg(nn.Module):
    """TCN Hybrid for regression — output is raw return (no sigmoid)."""

    def __init__(self, n_dynamic, n_static, n_filters=64, kernel_size=3,
                 n_blocks=4, head_hidden=64, dropout=0.2, window_size=14):
        super().__init__()
        n_blocks   = self._auto_n_blocks(n_blocks, kernel_size, window_size)
        self.n_blocks = n_blocks
        self.rf       = self._compute_rf(kernel_size, n_blocks)

        blocks = []
        for i in range(n_blocks):
            in_ch  = n_dynamic if i == 0 else n_filters
            blocks.append(TCNBlock(in_ch, n_filters, kernel_size, 2 ** i, dropout))
        self.tcn_stack = nn.Sequential(*blocks)

        head_in = n_filters + n_static
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),   # raw regression output
        )
        self._init_weights()

    @staticmethod
    def _compute_rf(kernel_size, n_blocks):
        return 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))

    @staticmethod
    def _auto_n_blocks(n_blocks, kernel_size, window_size):
        rf = 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))
        while rf < window_size and n_blocks < 8:
            n_blocks += 1
            rf = 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))
        return n_blocks

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_dyn, x_stat):
        feat     = self.tcn_stack(x_dyn)          # (N, n_filters, W)
        last     = feat[:, :, -1]                  # (N, n_filters)
        combined = torch.cat([last, x_stat], dim=1) if x_stat.shape[1] > 0 else last
        return self.head(combined).squeeze(1)      # (N,) raw prediction


# ─── Early Stopping (minimize MAE) ────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_mae   = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_mae, model):
        if val_mae < self.best_mae - self.min_delta:
            self.best_mae   = val_mae
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm):
    model.train()
    total_loss = 0.0
    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_dyn, x_stat), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        preds = model(x_dyn, x_stat)
        total_loss += criterion(preds, y).item() * len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    preds_np  = np.array(all_preds,  dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.float32)
    mae       = float(np.mean(np.abs(preds_np - labels_np)))
    return total_loss / len(loader.dataset), mae, preds_np, labels_np


def compute_full_metrics(name, y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r, _ = pearsonr(y_true, y_pred)
        r = float(r)
    except Exception:
        r = float("nan")

    signal     = (y_pred > SIGNAL_THR).astype(int)
    actual_up  = (y_true > SIGNAL_THR).astype(int)
    n_signals  = int(signal.sum())
    signal_acc = float((signal == actual_up).mean()) if len(actual_up) > 0 else float("nan")

    print(f"   [{name:<6}]  n={len(y_true):>4}  MAE={mae:.5f}  RMSE={rmse:.5f}"
          f"  r={r:.4f}  n_signals={n_signals}  signal_acc={signal_acc:.4f}")

    return {"split": name, "n_samples": int(len(y_true)),
            "mae": mae, "rmse": rmse,
            "pearson_r": r if not math.isnan(r) else None,
            "n_signals": n_signals,
            "signal_acc": signal_acc if not math.isnan(signal_acc) else None}


# ─── Tensor Loading ───────────────────────────────────────────────────────────

def load_tensors(scenario_dir):
    required = ["X_train_dynamic.npy", "X_val_dynamic.npy", "X_test_dynamic.npy",
                "y_train.npy", "y_val.npy", "y_test.npy"]
    if not os.path.isdir(scenario_dir):
        return None
    for f in required:
        if not os.path.exists(os.path.join(scenario_dir, f)):
            print(f"   [SKIP] Thiếu: {f}")
            return None

    def npy(name):
        return np.load(os.path.join(scenario_dir, name))

    def stat(split):
        p   = os.path.join(scenario_dir, f"X_{split}_static.npy")
        arr = np.load(p) if os.path.exists(p) else np.zeros((0, 0))
        return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    return dict(
        X_train_dyn  = npy("X_train_dynamic.npy"),
        X_val_dyn    = npy("X_val_dynamic.npy"),
        X_test_dyn   = npy("X_test_dynamic.npy"),
        X_train_stat = stat("train"),
        X_val_stat   = stat("val"),
        X_test_stat  = stat("test"),
        y_train      = npy("y_train.npy").astype(np.float32),
        y_val        = npy("y_val.npy").astype(np.float32),
        y_test       = npy("y_test.npy").astype(np.float32),
    )


# ─── Window Experiment ────────────────────────────────────────────────────────

def run_window_experiment(file_tag, window_size, tensors, batch_size, device, cfg):
    set_seed(cfg["seed"])

    X_tr_dyn, X_vl_dyn, X_te_dyn = tensors["X_train_dyn"], tensors["X_val_dyn"], tensors["X_test_dyn"]
    X_tr_st,  X_vl_st,  X_te_st  = tensors["X_train_stat"], tensors["X_val_stat"], tensors["X_test_stat"]
    y_tr,     y_vl,     y_te      = tensors["y_train"], tensors["y_val"], tensors["y_test"]

    if len(y_tr) == 0 or len(y_vl) == 0:
        print("   [SKIP] Tensor rỗng.")
        return None

    n_dynamic = X_tr_dyn.shape[2]
    n_static  = X_tr_st.shape[1] if X_tr_st.ndim == 2 else 0

    model = TCNHybridReg(
        n_dynamic   = n_dynamic,
        n_static    = n_static,
        n_filters   = cfg["n_filters"],
        kernel_size = cfg["kernel_size"],
        n_blocks    = cfg["n_blocks"],
        head_hidden = cfg["head_hidden"],
        dropout     = cfg["dropout"],
        window_size = window_size,
    ).to(device)

    rf       = model.rf
    n_blocks = model.n_blocks
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Window={window_size}  RF={rf}  blocks={n_blocks}"
          f"  dyn={n_dynamic}  static={n_static}"
          f"  n_train={len(y_tr)}  params={n_params:,}")

    train_loader = make_loader(X_tr_dyn, X_tr_st, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X_vl_dyn, X_vl_st, y_vl, batch_size)
    test_loader  = make_loader(X_te_dyn, X_te_st, y_te, batch_size)

    criterion  = nn.HuberLoss(delta=cfg["huber_delta"])
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"], min_lr=1e-5,
    )
    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"   {'Epoch':>5}  {'TrLoss':>9}  {'VaLoss':>9}  {'Val MAE':>9}  {'LR':>9}")
    for epoch in range(1, cfg["max_epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg["clip_norm"])
        va_loss, va_mae, _, _ = eval_epoch(model, val_loader, criterion, device)
        lr_now  = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"   {epoch:>5}  {tr_loss:>9.4f}  {va_loss:>9.4f}"
                  f"  {va_mae:>9.5f}  {lr_now:>9.2e}")

        scheduler.step(va_mae)
        if early_stop.step(va_mae, model):
            print(f"   Early stop tại epoch {epoch}. Best val MAE = {early_stop.best_mae:.5f}")
            break

    early_stop.restore_best(model)
    best_val_mae = early_stop.best_mae

    print(f"\n   Final metrics (best val MAE = {best_val_mae:.5f}):")
    _, _, vl_p, vl_l = eval_epoch(model, val_loader,   criterion, device)
    _, _, tr_p, tr_l = eval_epoch(model, train_loader, criterion, device)
    _, _, te_p, te_l = eval_epoch(model, test_loader,  criterion, device)

    results = {
        "train": compute_full_metrics("TRAIN", tr_l, tr_p),
        "val":   compute_full_metrics("VAL",   vl_l, vl_p),
        "test":  compute_full_metrics("TEST",  te_l, te_p),
    }

    return dict(
        window_size  = window_size,
        n_blocks     = n_blocks,
        rf           = rf,
        best_val_mae = best_val_mae,
        n_dynamic    = n_dynamic,
        n_static     = n_static,
        n_params     = n_params,
        results      = results,
        model        = model,
        test_preds   = te_p,
        test_labels  = te_l,
        val_preds    = vl_p,
        val_labels   = vl_l,
    )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(file_tag, is_daily):
    print(f"\n{'='*64}")
    print(f"  {file_tag.upper()} — TCN Regression")
    print(f"{'='*64}")

    batch_size = CFG["batch_daily"] if is_daily else CFG["batch_weekly"]
    device     = torch.device(CFG["device"])

    available_windows = []
    for win_dir in sorted(
        (d for d in os.listdir(TENSOR_DIR) if d.startswith("win_")),
        key=lambda d: int(d.split("_")[1]),
    ):
        scenario_dir = os.path.join(TENSOR_DIR, win_dir, file_tag)
        if os.path.isdir(scenario_dir):
            available_windows.append((int(win_dir.split("_")[1]), scenario_dir))

    if not available_windows:
        print(f"   [SKIP] Không tìm thấy tensors cho {file_tag}")
        return None

    print(f"   Window sizes: {[w for w, _ in available_windows]}")

    best_val_mae      = np.inf
    best_result       = None
    best_scenario_dir = None

    for w, scenario_dir in available_windows:
        print(f"\n  ── Window = {w} ──")
        tensors = load_tensors(scenario_dir)
        if tensors is None:
            continue
        result = run_window_experiment(file_tag, w, tensors, batch_size, device, CFG)
        if result is None:
            continue
        if result["best_val_mae"] < best_val_mae:
            best_val_mae      = result["best_val_mae"]
            best_result       = result
            best_scenario_dir = scenario_dir

    if best_result is None:
        print("   [ERROR] Không có window nào train thành công.")
        return None

    best_w = best_result["window_size"]
    print(f"\n  ── Best window: {best_w}  RF={best_result['rf']}"
          f"  val MAE={best_val_mae:.5f} ──")

    model      = best_result.pop("model")
    te_preds   = best_result.pop("test_preds")
    te_labels  = best_result.pop("test_labels")
    val_preds  = best_result.pop("val_preds")
    val_labels = best_result.pop("val_labels")

    model_path    = os.path.join(OUTPUT_DIR, f"tcn_reg_{file_tag}_win{best_w}.pt")
    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{file_tag}.csv")
    pred_path     = os.path.join(OUTPUT_DIR, f"test_predictions_{file_tag}.csv")
    result_path   = os.path.join(OUTPUT_DIR, f"results_{file_tag}.json")

    torch.save(model.state_dict(), model_path)

    val_meta_path = os.path.join(best_scenario_dir, "val_metadata.parquet")
    if os.path.exists(val_meta_path):
        val_meta  = pd.read_parquet(val_meta_path)
        val_dates = val_meta["Date"].values if "Date" in val_meta.columns else range(len(val_labels))
    else:
        val_dates = range(len(val_labels))
    pd.DataFrame({
        "Date":           val_dates,
        "y_true":         val_labels,
        "y_pred_tcn_reg": val_preds,
    }).to_csv(val_pred_path, index=False)

    pd.DataFrame({
        "y_true":         te_labels,
        "y_pred_tcn_reg": te_preds,
    }).to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "file_tag":    file_tag,
            "is_daily":    is_daily,
            "best_window": best_w,
            "best_val_mae": float(best_val_mae),
            "n_blocks":    best_result["n_blocks"],
            "rf":          best_result["rf"],
            "n_params":    best_result["n_params"],
            "cfg":         {k: v for k, v in CFG.items() if k != "device"},
            "splits":      best_result["results"],
        }, f, indent=2)

    print(f"   Saved: {os.path.basename(model_path)}")
    return best_result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {CFG['device']}")
    if CFG["device"] == "cpu":
        print("[INFO] CPU mode — TCN train nhanh hơn LSTM vì fully parallelizable.")

    all_results = {}

    for tag in FILES_DAILY:
        try:
            res = run_pipeline(tag, is_daily=True)
            if res:
                all_results[tag] = res
        except Exception as e:
            print(f"\n[ERROR] {tag}: {e}")
            import traceback; traceback.print_exc()

    for tag in FILES_WEEKLY:
        try:
            res = run_pipeline(tag, is_daily=False)
            if res:
                all_results[tag] = res
        except Exception as e:
            print(f"\n[ERROR] {tag}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        print(f"\n\n{'='*64}")
        print("  SUMMARY — TCN Regression Best Window")
        print(f"{'='*64}")
        print(f"  {'Tag':<34} {'Win':>4} {'RF':>5} {'Val MAE':>9} {'Test MAE':>9} {'Pearson r':>10}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            t    = res["results"].get("test", {})
            name = tag.replace("integrated_", "")
            print(
                f"  {name:<34}"
                f" {res['window_size']:>4}"
                f" {res['rf']:>5}"
                f" {res['best_val_mae']:>9.5f}"
                f" {t.get('mae', 0.0):>9.5f}"
                f" {t.get('pearson_r') or 0.0:>10.4f}"
            )
        print(f"\n  Predictions saved to {OUTPUT_DIR}")
