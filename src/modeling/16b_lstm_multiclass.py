"""
16b_lstm_multiclass.py — Tầng 2b: LSTM Hybrid Multiclass (down/flat/up)
========================================================================
Giống 16_lstm_hybrid.py nhưng:
  - Load tensors từ data/tensors_multiclass_soft/ (y shape N×3 soft labels)
  - Loss: KLDivLoss(log_softmax(logits), y_soft)
  - Head output: Linear(head_hidden, 3) → 3 logits
  - Val AUC: macro OvR (argmax của y_soft làm hard labels)
  - Lưu 3 prob columns: y_prob_lstm_{down,flat,up}
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors_multiclass_soft")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "16b_lstm_multiclass")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLASSES   = 3
CLASS_NAMES = ["down", "flat", "up"]

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
    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)

FILES_DAILY  = ["integrated_coffee_daily",  "integrated_corn_daily"]
FILES_WEEKLY = ["integrated_coffee_weekly", "integrated_corn_weekly"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, X_dyn, X_stat, y):
        self.X_dyn  = torch.FloatTensor(X_dyn)
        self.X_stat = torch.FloatTensor(X_stat) \
                      if X_stat.ndim == 2 and X_stat.shape[1] > 0 \
                      else torch.zeros(len(X_dyn), 0)
        # y shape: (N, 3) soft distributions
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_dyn[idx], self.X_stat[idx], self.y[idx]


def make_loader(X_dyn, X_stat, y, batch_size, shuffle=False):
    ds = HybridDataset(X_dyn, X_stat, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMHybridMC(nn.Module):
    """
    Giống LSTMHybrid nhưng head output N_CLASSES logits (không phải 1).
    Loss: KLDivLoss với log_softmax(logits) và soft-label targets.
    """

    def __init__(self, n_dynamic, n_static, h1=128, h2=64, head_hidden=64,
                 dropout=0.3, bidirectional=False, n_classes=3):
        super().__init__()
        dir_mult = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(
            input_size=n_dynamic, hidden_size=h1,
            num_layers=1, batch_first=True, bidirectional=bidirectional,
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=h1 * dir_mult, hidden_size=h2,
            num_layers=1, batch_first=True,
        )
        self.drop2 = nn.Dropout(dropout)

        head_in = h2 + n_static
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),  # (N, 3) logits
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
        return self.head(combined)   # (N, 3) logits


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_auc   = -np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_auc, model):
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc   = val_auc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm, class_w=None):
    model.train()
    total_loss = 0.0
    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_dyn, x_stat)          # (N, 3)
        log_p  = F.log_softmax(logits, dim=1)  # (N, 3)

        if class_w is not None:
            # Per-sample weighted KL loss — penalizes Up/Down mispredictions more
            per_sample = F.kl_div(log_p, y, reduction="none").sum(dim=1)  # (N,)
            w = class_w.to(device)[torch.argmax(y, dim=1)]
            loss = (per_sample * w).mean()
        else:
            loss = criterion(log_p, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Returns (avg_loss, macro_ovr_auc, probs (N,3), hard_labels (N,))."""
    model.eval()
    all_probs, all_hard = [], []
    total_loss = 0.0

    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        logits  = model(x_dyn, x_stat)
        log_p   = F.log_softmax(logits, dim=1)
        total_loss += criterion(log_p, y).item() * len(y)

        probs = F.softmax(logits, dim=1).cpu().numpy()      # (N, 3)
        hard  = np.argmax(y.cpu().numpy(), axis=1)          # argmax of soft targets
        all_probs.append(probs)
        all_hard.extend(hard.tolist())

    probs_np = np.concatenate(all_probs, axis=0).astype(np.float32)  # (N, 3)
    hard_np  = np.array(all_hard, dtype=np.int64)

    try:
        auc = float(roc_auc_score(hard_np, probs_np, multi_class="ovr", average="macro")) \
              if len(np.unique(hard_np)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return total_loss / len(loader.dataset), auc, probs_np, hard_np


def compute_full_metrics(name, y_hard, probs):
    y_pred = np.argmax(probs, axis=1)
    try:
        auc = float(roc_auc_score(y_hard, probs, multi_class="ovr", average="macro")) \
              if len(np.unique(y_hard)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    f1  = float(f1_score(y_hard, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_hard, y_pred))
    dist = {int(c): int((y_hard == c).sum()) for c in sorted(np.unique(y_hard))}

    print(
        f"   [{name:<6}]  n={len(y_hard):>4}  dist={dist}"
        f"  OvR-AUC={auc:.4f}  MacroF1={f1:.4f}  Acc={acc:.4f}"
    )
    return {
        "split":     name,
        "n_samples": int(len(y_hard)),
        "dist":      dist,
        "auc_roc":   auc if not np.isnan(auc) else None,
        "f1":        f1,
        "accuracy":  acc,
    }


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

    y_tr = npy("y_train.npy").astype(np.float32)
    if y_tr.ndim == 1 or y_tr.shape[1] != 3:
        print(f"   [SKIP] y_train shape={y_tr.shape} — cần (N, 3). Chạy 13b trước.")
        return None

    return dict(
        X_train_dyn  = npy("X_train_dynamic.npy"),
        X_val_dyn    = npy("X_val_dynamic.npy"),
        X_test_dyn   = npy("X_test_dynamic.npy"),
        X_train_stat = stat("train"),
        X_val_stat   = stat("val"),
        X_test_stat  = stat("test"),
        y_train      = y_tr,
        y_val        = npy("y_val.npy").astype(np.float32),
        y_test       = npy("y_test.npy").astype(np.float32),
    )


# ─── Experiment Runner ────────────────────────────────────────────────────────

def run_window_experiment(file_tag, window_size, tensors, batch_size, device, cfg):
    set_seed(cfg["seed"])

    X_tr_dyn, X_vl_dyn, X_te_dyn = tensors["X_train_dyn"], tensors["X_val_dyn"], tensors["X_test_dyn"]
    X_tr_st,  X_vl_st,  X_te_st  = tensors["X_train_stat"], tensors["X_val_stat"], tensors["X_test_stat"]
    y_tr, y_vl, y_te = tensors["y_train"], tensors["y_val"], tensors["y_test"]

    if len(y_tr) == 0 or len(y_vl) == 0:
        print("   [SKIP] Tensor rỗng.")
        return None

    n_dynamic = X_tr_dyn.shape[2]
    n_static  = X_tr_st.shape[1] if X_tr_st.ndim == 2 else 0

    print(f"   Window={window_size}  dyn={n_dynamic}  static={n_static}"
          f"  n_train={len(y_tr)}  n_val={len(y_vl)}")

    train_loader = make_loader(X_tr_dyn, X_tr_st, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X_vl_dyn, X_vl_st, y_vl, batch_size)
    test_loader  = make_loader(X_te_dyn, X_te_st, y_te, batch_size)

    model = LSTMHybridMC(
        n_dynamic=n_dynamic, n_static=n_static,
        h1=cfg["h1"], h2=cfg["h2"], head_hidden=cfg["head_hidden"],
        dropout=cfg["dropout"], bidirectional=cfg["bidirectional"],
        n_classes=N_CLASSES,
    ).to(device)

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["lr_patience"], min_lr=1e-5,
    )
    early_stop = EarlyStopping(patience=cfg["patience"])

    # Inverse-frequency class weights from soft-label argmax — boost Up/Down learning
    y_hard_tr  = np.argmax(y_tr, axis=1)
    counts     = np.bincount(y_hard_tr, minlength=3).clip(1)
    class_w    = torch.tensor(len(y_hard_tr) / (3 * counts), dtype=torch.float32)

    print(f"   {'Epoch':>5}  {'TrLoss':>9}  {'VaLoss':>9}  {'Val AUC':>9}  {'LR':>9}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg["clip_norm"], class_w=class_w)
        va_loss, va_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        lr_now  = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"   {epoch:>5}  {tr_loss:>9.4f}  {va_loss:>9.4f}  {va_auc:>9.4f}  {lr_now:>9.2e}")

        scheduler.step(va_auc)
        if early_stop.step(va_auc, model):
            print(f"   Early stop tại epoch {epoch}. Best val AUC = {early_stop.best_auc:.4f}")
            break

    early_stop.restore_best(model)

    print(f"\n   Final metrics (best val AUC = {early_stop.best_auc:.4f}):")
    _, _, vl_p, vl_h = eval_epoch(model, val_loader,   criterion, device)
    _, _, tr_p, tr_h = eval_epoch(model, train_loader,  criterion, device)
    _, _, te_p, te_h = eval_epoch(model, test_loader,   criterion, device)

    results = {
        "train": compute_full_metrics("TRAIN", tr_h, tr_p),
        "val":   compute_full_metrics("VAL",   vl_h, vl_p),
        "test":  compute_full_metrics("TEST",  te_h, te_p),
    }

    return dict(
        window_size   = window_size,
        best_val_auc  = early_stop.best_auc,
        n_dynamic     = n_dynamic,
        n_static      = n_static,
        results       = results,
        model         = model,
        test_probs    = te_p,    # (N, 3)
        test_labels   = te_h,    # (N,) hard
        val_probs     = vl_p,
        val_labels    = vl_h,
    )


def run_pipeline(file_tag, is_daily):
    print(f"\n{'═'*64}")
    print(f"  {file_tag.upper()} — LSTM Multiclass")
    print(f"{'═'*64}")

    batch_size = CFG["batch_daily"] if is_daily else CFG["batch_weekly"]
    device     = torch.device(CFG["device"])

    if not os.path.isdir(TENSOR_DIR):
        print(f"   [SKIP] Không tìm thấy {TENSOR_DIR}. Chạy 13b_tensor_packing_mc.py trước.")
        return None

    available_windows = []
    for win_dir in sorted(
        (d for d in os.listdir(TENSOR_DIR) if d.startswith("win_")),
        key=lambda d: int(d.split("_")[1]),
    ):
        scenario_dir = os.path.join(TENSOR_DIR, win_dir, file_tag)
        if os.path.isdir(scenario_dir):
            w = int(win_dir.split("_")[1])
            available_windows.append((w, scenario_dir))

    if not available_windows:
        print(f"   [SKIP] Không tìm thấy tensors cho {file_tag} trong {TENSOR_DIR}")
        return None

    print(f"   Windows: {[w for w, _ in available_windows]}")

    best_val_auc      = -np.inf
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
        if result["best_val_auc"] > best_val_auc:
            best_val_auc      = result["best_val_auc"]
            best_result       = result
            best_scenario_dir = scenario_dir

    if best_result is None:
        print("   [ERROR] Không có window nào train thành công.")
        return None

    best_w   = best_result["window_size"]
    print(f"\n  ── Best window: {best_w}  val AUC={best_val_auc:.4f} ──")

    model      = best_result.pop("model")
    te_probs   = best_result.pop("test_probs")    # (N, 3)
    te_labels  = best_result.pop("test_labels")   # (N,) hard
    val_probs  = best_result.pop("val_probs")
    val_labels = best_result.pop("val_labels")

    model_path    = os.path.join(OUTPUT_DIR, f"lstm_{file_tag}_win{best_w}.pt")
    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{file_tag}.csv")
    pred_path     = os.path.join(OUTPUT_DIR, f"test_predictions_{file_tag}.csv")
    result_path   = os.path.join(OUTPUT_DIR, f"results_{file_tag}.json")

    torch.save(model.state_dict(), model_path)

    # Val predictions — attach Date từ val_metadata.parquet
    val_meta_path = os.path.join(best_scenario_dir, "val_metadata.parquet")
    if os.path.exists(val_meta_path):
        val_meta  = pd.read_parquet(val_meta_path)
        val_dates = val_meta["Date"].values if "Date" in val_meta.columns else range(len(val_labels))
    else:
        val_dates = range(len(val_labels))

    val_pred_df = pd.DataFrame({"Date": val_dates, "y_true": val_labels})
    for i, cls in enumerate(CLASS_NAMES):
        val_pred_df[f"y_prob_lstm_{cls}"] = val_probs[:, i]
    val_pred_df.to_csv(val_pred_path, index=False)

    # Test predictions
    test_pred_df = pd.DataFrame({
        "y_true":      te_labels,
        "y_pred_lstm": np.argmax(te_probs, axis=1),
    })
    for i, cls in enumerate(CLASS_NAMES):
        test_pred_df[f"y_prob_lstm_{cls}"] = te_probs[:, i]
    test_pred_df.to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "file_tag":     file_tag,
            "is_daily":     is_daily,
            "best_window":  best_w,
            "best_val_auc": float(best_val_auc),
            "n_dynamic":    best_result["n_dynamic"],
            "n_static":     best_result["n_static"],
            "cfg":          {k: v for k, v in CFG.items() if k != "device"},
            "splits":       best_result["results"],
        }, f, indent=2)

    print(f"   Saved: {os.path.basename(model_path)}")
    return best_result


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
        print(f"\n\n{'═'*64}")
        print("  SUMMARY — LSTM Multiclass Best Window")
        print(f"{'═'*64}")
        print(f"  {'Tag':<38} {'Win':>4} {'Val AUC':>9} {'Test AUC':>9} {'F1-mac':>8}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            t    = res["results"].get("test", {})
            name = tag.replace("integrated_", "")
            print(
                f"  {name:<38}"
                f" {res['window_size']:>4}"
                f" {res['best_val_auc']:>9.4f}"
                f" {t.get('auc_roc') or 0.0:>9.4f}"
                f" {t.get('f1', 0.0):>8.4f}"
            )
        print(f"\n  Predictions saved to {OUTPUT_DIR}")
