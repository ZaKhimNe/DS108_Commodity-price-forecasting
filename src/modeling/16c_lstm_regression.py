"""
16c_lstm_regression.py — Tầng 2c: LSTM Regression
===================================================
Load tensors từ data/tensors_reg/win_{W}/{file_tag}/ (output của 13c_tensor_packing_reg.py).
Dùng HuberLoss thay BCEWithLogitsLoss; output là raw return prediction (không sigmoid).
Best window chọn theo val MAE thấp nhất (không phải AUC).

Chạy:
    python src/modeling/16c_lstm_regression.py
"""

import os
import json
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "16c_lstm_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = dict(
    h1            = 128,
    h2            = 64,
    head_hidden   = 64,
    dropout       = 0.30,
    bidirectional = False,

    lr            = 1e-3,
    weight_decay  = 1e-4,
    max_epochs    = 120,
    batch_daily   = 64,
    batch_weekly  = 32,
    patience      = 15,
    lr_patience   = 5,
    clip_norm     = 1.0,
    huber_delta   = 0.05,   # HuberLoss delta — target range ~[-0.30, 0.30]

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
        self.X_dyn  = torch.FloatTensor(X_dyn)
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


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMHybridReg(nn.Module):
    """
    LSTM Hybrid for regression.
    Output: raw scalar (no sigmoid) — predicted return_future.
    """

    def __init__(self, n_dynamic, n_static, h1=128, h2=64, head_hidden=64,
                 dropout=0.3, bidirectional=False):
        super().__init__()
        dir_mult = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size=n_dynamic, hidden_size=h1,
                             num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=h1 * dir_mult, hidden_size=h2,
                             num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        head_in = h2 + n_static
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),  # raw regression output
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x_dyn, x_stat):
        out1, _ = self.lstm1(x_dyn)
        out1    = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        h_T     = self.drop2(out2[:, -1, :])

        combined = torch.cat([h_T, x_stat], dim=1) if x_stat.shape[1] > 0 else h_T
        return self.head(combined).squeeze(1)   # (N,) raw prediction


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
        preds = model(x_dyn, x_stat)
        loss  = criterion(preds, y)
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
    avg_loss  = total_loss / len(loader.dataset)
    return avg_loss, mae, preds_np, labels_np


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
            "pearson_r": r if not np.isnan(r) else None,
            "n_signals": n_signals,
            "signal_acc": signal_acc if not np.isnan(signal_acc) else None}


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

    X_tr_dyn  = tensors["X_train_dyn"]
    X_vl_dyn  = tensors["X_val_dyn"]
    X_te_dyn  = tensors["X_test_dyn"]
    X_tr_stat = tensors["X_train_stat"]
    X_vl_stat = tensors["X_val_stat"]
    X_te_stat = tensors["X_test_stat"]
    y_tr      = tensors["y_train"]
    y_vl      = tensors["y_val"]
    y_te      = tensors["y_test"]

    if len(y_tr) == 0 or len(y_vl) == 0:
        print("   [SKIP] Tensor rỗng.")
        return None

    n_dynamic = X_tr_dyn.shape[2]
    n_static  = X_tr_stat.shape[1] if X_tr_stat.ndim == 2 else 0

    print(f"   Window={window_size}  dyn={n_dynamic}  static={n_static}"
          f"  n_train={len(y_tr)}  n_val={len(y_vl)}")

    train_loader = make_loader(X_tr_dyn, X_tr_stat, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X_vl_dyn, X_vl_stat, y_vl, batch_size)
    test_loader  = make_loader(X_te_dyn, X_te_stat, y_te, batch_size)

    model = LSTMHybridReg(
        n_dynamic     = n_dynamic,
        n_static      = n_static,
        h1            = cfg["h1"],
        h2            = cfg["h2"],
        head_hidden   = cfg["head_hidden"],
        dropout       = cfg["dropout"],
        bidirectional = cfg["bidirectional"],
    ).to(device)

    criterion = nn.HuberLoss(delta=cfg["huber_delta"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["lr_patience"], min_lr=1e-5,
    )
    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"   {'Epoch':>5}  {'TrLoss':>9}  {'VaLoss':>9}  {'Val MAE':>9}  {'LR':>9}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg["clip_norm"])
        val_loss, val_mae, _, _ = eval_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"   {epoch:>5}  {train_loss:>9.4f}  {val_loss:>9.4f}"
                  f"  {val_mae:>9.5f}  {current_lr:>9.2e}")

        scheduler.step(val_mae)

        if early_stop.step(val_mae, model):
            print(f"   Early stop tại epoch {epoch}. Best val MAE = {early_stop.best_mae:.5f}")
            break

    early_stop.restore_best(model)
    best_val_mae = early_stop.best_mae

    print(f"\n   Final metrics (best val MAE = {best_val_mae:.5f}):")
    _, _, val_preds, val_labels = eval_epoch(model, val_loader,   criterion, device)
    _, _, tr_preds,  tr_labels  = eval_epoch(model, train_loader, criterion, device)
    _, _, te_preds,  te_labels  = eval_epoch(model, test_loader,  criterion, device)

    results = {
        "train": compute_full_metrics("TRAIN", tr_labels, tr_preds),
        "val":   compute_full_metrics("VAL",   val_labels, val_preds),
        "test":  compute_full_metrics("TEST",  te_labels, te_preds),
    }

    return {
        "window_size":  window_size,
        "best_val_mae": best_val_mae,
        "n_dynamic":    n_dynamic,
        "n_static":     n_static,
        "results":      results,
        "model":        model,
        "test_preds":   te_preds,
        "test_labels":  te_labels,
        "val_preds":    val_preds,
        "val_labels":   val_labels,
    }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(file_tag, is_daily):
    print(f"\n{'='*64}")
    print(f"  {file_tag.upper()} — LSTM Regression")
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
        print(f"   [SKIP] Không tìm thấy tensors cho {file_tag} trong {TENSOR_DIR}")
        return None

    print(f"   Windows: {[w for w, _ in available_windows]}")

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
    print(f"\n  ── Best window: {best_w}  val MAE={best_val_mae:.5f} ──")

    model      = best_result.pop("model")
    te_preds   = best_result.pop("test_preds")
    te_labels  = best_result.pop("test_labels")
    val_preds  = best_result.pop("val_preds")
    val_labels = best_result.pop("val_labels")

    model_path    = os.path.join(OUTPUT_DIR, f"lstm_reg_{file_tag}_win{best_w}.pt")
    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{file_tag}.csv")
    pred_path     = os.path.join(OUTPUT_DIR, f"test_predictions_{file_tag}.csv")
    result_path   = os.path.join(OUTPUT_DIR, f"results_{file_tag}.json")

    torch.save(model.state_dict(), model_path)

    # Val predictions — attach Date from val_metadata.parquet
    val_meta_path = os.path.join(best_scenario_dir, "val_metadata.parquet")
    if os.path.exists(val_meta_path):
        val_meta  = pd.read_parquet(val_meta_path)
        val_dates = val_meta["Date"].values if "Date" in val_meta.columns else range(len(val_labels))
    else:
        val_dates = range(len(val_labels))
    pd.DataFrame({
        "Date":            val_dates,
        "y_true":          val_labels,
        "y_pred_lstm_reg": val_preds,
    }).to_csv(val_pred_path, index=False)

    # Test predictions (no Date — stacking 18c attaches from test_metadata.parquet)
    pd.DataFrame({
        "y_true":          te_labels,
        "y_pred_lstm_reg": te_preds,
    }).to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "file_tag":    file_tag,
            "is_daily":    is_daily,
            "best_window": best_w,
            "best_val_mae": float(best_val_mae),
            "n_dynamic":   best_result["n_dynamic"],
            "n_static":    best_result["n_static"],
            "cfg":         {k: v for k, v in CFG.items() if k != "device"},
            "splits":      best_result["results"],
        }, f, indent=2)

    print(f"   Saved: {os.path.basename(model_path)}")
    return best_result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {CFG['device']}")

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
        print("  SUMMARY — LSTM Regression Best Window")
        print(f"{'='*64}")
        print(f"  {'Tag':<38} {'Win':>4} {'Val MAE':>9} {'Test MAE':>9} {'Pearson r':>10}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            t    = res["results"].get("test", {})
            name = tag.replace("integrated_", "")
            print(
                f"  {name:<38}"
                f" {res['window_size']:>4}"
                f" {res['best_val_mae']:>9.5f}"
                f" {t.get('mae', 0.0):>9.5f}"
                f" {t.get('pearson_r') or 0.0:>10.4f}"
            )
        print(f"\n  Predictions saved to {OUTPUT_DIR}")
