"""
lstm_hybrid.py — Tầng 2: LSTM Hybrid (Dynamic 3D + Static 2D)
==============================================================
Load tensors từ data/tensors/win_{W}/{file_tag}/ (output của tensor_packing.py),
train LSTM Hybrid với dynamic sequence encoder + static feature injection.

Thứ tự ưu tiên:
    1. Chạy lgbm_baseline.py → lấy floor AUC
    2. Chạy script này → thử từng window size, pick best val AUC
    3. So sánh test AUC với LightGBM — nếu không thắng, xem lại feature engineering

Chạy:
    python src/modeling/lstm_hybrid.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "16_lstm_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = dict(
    # Model
    h1            = 128,    # First LSTM hidden size
    h2            = 64,     # Second LSTM hidden size (h_T dimension)
    head_hidden   = 64,     # Dense head hidden size
    dropout       = 0.30,
    bidirectional = False,  # True → BiLSTM trên layer đầu (thử sau LSTM baseline)

    # Training
    lr            = 1e-3,
    weight_decay  = 1e-4,
    max_epochs    = 120,
    batch_daily   = 64,
    batch_weekly  = 32,
    patience      = 15,     # Early stopping patience (epochs)
    lr_patience   = 5,      # ReduceLROnPlateau patience
    clip_norm     = 1.0,    # Gradient clipping
    max_spw       = 10.0,   # Cap scale_pos_weight để tránh training mất ổn định

    # Misc
    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)

FILES_DAILY  = ["integrated_coffee_daily",  "integrated_corn_daily"]
FILES_WEEKLY = ["integrated_coffee_weekly", "integrated_corn_weekly"]

# v5: expand to {14,30,45} — LSTM can now pick short window if val AUC is better
# TCN is restricted to {14,30} → still maintains temporal scale diversity
ALLOWED_WINDOWS_DAILY  = {14, 30, 45}
ALLOWED_WINDOWS_WEEKLY = None   # weekly has few windows — no restriction


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    """
    Wrapper cho cặp (dynamic, static, label).
    static có thể là array rỗng (shape (N, 0)) nếu không có static features.
    """

    def __init__(self, X_dyn: np.ndarray, X_stat: np.ndarray, y: np.ndarray):
        self.X_dyn  = torch.FloatTensor(X_dyn)
        self.X_stat = torch.FloatTensor(X_stat) if X_stat.ndim == 2 and X_stat.shape[1] > 0 \
                      else torch.zeros(len(X_dyn), 0)
        self.y      = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_dyn[idx], self.X_stat[idx], self.y[idx]


def make_loader(
    X_dyn: np.ndarray,
    X_stat: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    ds = HybridDataset(X_dyn, X_stat, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMHybrid(nn.Module):
    """
    Architecture:
        Dynamic (N, W, D) ──► LSTM(h1) ──► LSTM(h2) ──► h_T
                                                             │
        Static  (N, S)   ─────────────────────────────────► Concat
                                                             │
                                                         BN + Dense(64) + Dropout
                                                             │
                                                         Dense(1)  [logits]

    Lưu ý:
    - Output là LOGITS, không phải probabilities.
      Dùng torch.sigmoid() khi cần probs ở inference.
    - BCEWithLogitsLoss tích hợp sigmoid + BCE cho numerical stability.
    - Nếu bidirectional=True: layer 1 là BiLSTM (2×h1), layer 2 là LSTM(h2).
      BiLSTM capture pattern ở đầu window tốt hơn vì LSTM đọc xuôi + ngược
      trong lịch sử (không phải look-ahead — toàn window đều là quá khứ).
    """

    def __init__(
        self,
        n_dynamic:    int,
        n_static:     int,
        h1:           int   = 128,
        h2:           int   = 64,
        head_hidden:  int   = 64,
        dropout:      float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        dir_mult = 2 if bidirectional else 1

        # Layer 1: optional BiLSTM
        self.lstm1 = nn.LSTM(
            input_size    = n_dynamic,
            hidden_size   = h1,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = bidirectional,
        )
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: unidirectional (luôn), nhận output của layer 1
        self.lstm2 = nn.LSTM(
            input_size  = h1 * dir_mult,
            hidden_size = h2,
            num_layers  = 1,
            batch_first = True,
        )
        self.drop2 = nn.Dropout(dropout)

        # Classification head: concat h_T với static features
        head_in = h2 + n_static
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),  # logits
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x_dyn:  torch.Tensor,   # (N, W, D)
        x_stat: torch.Tensor,   # (N, S)  — S=0 nếu không có static
    ) -> torch.Tensor:           # (N,) logits

        out1, _ = self.lstm1(x_dyn)             # (N, W, h1 * dir_mult)
        out1    = self.drop1(out1)

        out2, _ = self.lstm2(out1)              # (N, W, h2)
        h_T     = self.drop2(out2[:, -1, :])   # (N, h2)

        if x_stat.shape[1] > 0:
            combined = torch.cat([h_T, x_stat], dim=1)  # (N, h2+S)
        else:
            combined = h_T

        return self.head(combined).squeeze(1)   # (N,) logits


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Dừng training khi val AUC không cải thiện sau `patience` epochs."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_auc   = -np.inf
        self.counter    = 0
        self.best_state = None  # lưu model state tốt nhất trong RAM

    def step(self, val_auc: float, model: nn.Module) -> bool:
        """Returns True nếu nên dừng training."""
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc   = val_auc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Khôi phục weights của epoch tốt nhất."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    clip_norm: float = 1.0,
) -> float:
    """Train một epoch, trả về average loss."""
    model.train()
    total_loss = 0.0

    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_dyn, x_stat)
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping — tránh exploding gradient với LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model, trả về (avg_loss, auc_roc, probs, labels)."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)

        logits = model(x_dyn, x_stat)
        loss   = criterion(logits, y)
        probs  = torch.sigmoid(logits)

        total_loss  += loss.item() * len(y)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    probs_np  = np.array(all_probs,  dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.float32)

    auc      = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.5
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc, probs_np, labels_np


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tìm threshold tối ưu (maximize F1) trên validation set."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def compute_full_metrics(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Tính và in đầy đủ metrics cho một split."""
    y_pred    = (y_prob >= threshold).astype(int)
    has_both  = len(np.unique(y_true)) > 1

    auc_roc   = float(roc_auc_score(y_true, y_prob))            if has_both else float("nan")
    pr_auc    = float(average_precision_score(y_true, y_prob))  if has_both else float("nan")
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    prec      = float(precision_score(y_true, y_pred, zero_division=0))
    base_rate = float(y_true.mean())
    lift      = (prec / base_rate) if base_rate > 0 else float("nan")

    print(
        f"   [{name:<6}]  n={len(y_true):>4}  base={base_rate:.3f}  thr={threshold:.2f}"
        f"  AUC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}  Lift={lift:.2f}x"
    )

    return {
        "split":      name,
        "n_samples":  int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate":  base_rate,
        "threshold":  threshold,
        "auc_roc":    auc_roc if not np.isnan(auc_roc) else None,
        "pr_auc":     pr_auc  if not np.isnan(pr_auc)  else None,
        "f1":         f1,
        "precision":  prec,
        "lift":       lift    if not np.isnan(lift)     else None,
    }


# ─── Tensor Loading ───────────────────────────────────────────────────────────

def load_tensors(scenario_dir: str) -> dict | None:
    """
    Load tất cả .npy files cho một scenario (window_size × file_tag).
    Returns None nếu thư mục không tồn tại hoặc thiếu file bắt buộc.
    """
    required = [
        "X_train_dynamic.npy", "X_val_dynamic.npy", "X_test_dynamic.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
    ]

    if not os.path.isdir(scenario_dir):
        return None

    for f in required:
        if not os.path.exists(os.path.join(scenario_dir, f)):
            print(f"   [SKIP] Thiếu file: {f} trong {scenario_dir}")
            return None

    def load(name: str) -> np.ndarray:
        return np.load(os.path.join(scenario_dir, name))

    def load_static(split: str) -> np.ndarray:
        path = os.path.join(scenario_dir, f"X_{split}_static.npy")
        arr  = np.load(path) if os.path.exists(path) else np.zeros((0, 0))
        return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    return {
        "X_train_dyn":  load("X_train_dynamic.npy"),
        "X_val_dyn":    load("X_val_dynamic.npy"),
        "X_test_dyn":   load("X_test_dynamic.npy"),
        "X_train_stat": load_static("train"),
        "X_val_stat":   load_static("val"),
        "X_test_stat":  load_static("test"),
        "y_train":      load("y_train.npy").astype(np.float32),
        "y_val":        load("y_val.npy").astype(np.float32),
        "y_test":       load("y_test.npy").astype(np.float32),
    }


# ─── Experiment Runner ────────────────────────────────────────────────────────

def run_window_experiment(
    file_tag:    str,
    window_size: int,
    tensors:     dict,
    batch_size:  int,
    device:      torch.device,
    cfg:         dict,
) -> dict | None:
    """
    Train và evaluate LSTM Hybrid cho một (file_tag, window_size).
    Returns dict kết quả, hoặc None nếu data rỗng.
    """
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

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = float(min(neg / pos, cfg["max_spw"])) if pos > 0 else 1.0

    print(f"   Window={window_size}  dyn={n_dynamic}  static={n_static}"
          f"  n_train={len(y_tr)}  base={y_tr.mean():.3f}  spw={spw:.2f}")

    train_loader = make_loader(X_tr_dyn, X_tr_stat, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X_vl_dyn, X_vl_stat, y_vl, batch_size)
    test_loader  = make_loader(X_te_dyn, X_te_stat, y_te, batch_size)

    model = LSTMHybrid(
        n_dynamic     = n_dynamic,
        n_static      = n_static,
        h1            = cfg["h1"],
        h2            = cfg["h2"],
        head_hidden   = cfg["head_hidden"],
        dropout       = cfg["dropout"],
        bidirectional = cfg["bidirectional"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([spw], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",
        factor   = 0.5,
        patience = cfg["lr_patience"],
        min_lr   = 1e-5,
    )

    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"   Epoch  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>9}  {'LR':>9}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg["clip_norm"]
        )
        val_loss, val_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"   {epoch:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}"
                  f"  {val_auc:>9.4f}  {current_lr:>9.2e}")

        scheduler.step(val_auc)

        if early_stop.step(val_auc, model):
            print(f"   Early stop tại epoch {epoch}. Best val AUC = {early_stop.best_auc:.4f}")
            break

    # Restore best checkpoint trước khi evaluate
    early_stop.restore_best(model)
    best_val_auc = early_stop.best_auc

    print(f"\n   Final metrics (best val AUC = {best_val_auc:.4f}):")

    _, _, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
    _, _, tr_probs,  tr_labels  = eval_epoch(model, train_loader, criterion, device)
    _, _, te_probs,  te_labels  = eval_epoch(model, test_loader,  criterion, device)

    # Threshold tuning chỉ trên val
    best_thr = find_optimal_threshold(val_labels, val_probs)

    results = {
        "train": compute_full_metrics("TRAIN", tr_labels, tr_probs, best_thr),
        "val":   compute_full_metrics("VAL",   val_labels, val_probs, best_thr),
        "test":  compute_full_metrics("TEST",  te_labels, te_probs, best_thr),
    }

    train_auc = results["train"]["auc_roc"] or 0
    val_auc_f = results["val"]["auc_roc"]   or 0
    gap = train_auc - val_auc_f
    if gap > 0.12:
        print(f"   [WARN] Overfit gap={gap:.4f}: tăng dropout hoặc weight_decay.")

    return {
        "window_size":    window_size,
        "best_val_auc":   best_val_auc,
        "best_threshold": best_thr,
        "n_dynamic":      n_dynamic,
        "n_static":       n_static,
        "results":        results,
        "model":          model,
        "test_probs":     te_probs,
        "test_labels":    te_labels,
        "val_probs":      val_probs,
        "val_labels":     val_labels,
    }


def run_pipeline(file_tag: str, is_daily: bool) -> dict | None:
    """
    Thử tất cả window sizes cho một file_tag.
    Chọn window size có val AUC tốt nhất, lưu model và predictions.
    """
    print(f"\n{'═'*64}")
    print(f"  {file_tag.upper()}")
    print(f"{'═'*64}")

    batch_size = CFG["batch_daily"] if is_daily else CFG["batch_weekly"]
    device     = torch.device(CFG["device"])

    allowed = ALLOWED_WINDOWS_DAILY if is_daily else ALLOWED_WINDOWS_WEEKLY
    available_windows = []
    for win_dir in sorted(
        (d for d in os.listdir(TENSOR_DIR) if d.startswith("win_")),
        key=lambda d: int(d.split("_")[1]),
    ):
        scenario_dir = os.path.join(TENSOR_DIR, win_dir, file_tag)
        if os.path.isdir(scenario_dir):
            w = int(win_dir.split("_")[1])
            if allowed is not None and w not in allowed:
                continue
            available_windows.append((w, scenario_dir))

    if not available_windows:
        print(f"   [SKIP] Không tìm thấy tensors cho {file_tag} trong {TENSOR_DIR}")
        return None

    print(f"   Tìm thấy {len(available_windows)} window sizes: "
          f"{[w for w, _ in available_windows]}")

    best_val_auc     = -np.inf
    best_result      = None
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
        print(f"\n   [ERROR] Không có window size nào train thành công.")
        return None

    best_w = best_result["window_size"]
    print(f"\n  ── Best window size: {best_w}  (val AUC = {best_val_auc:.4f}) ──")

    model      = best_result.pop("model")
    te_probs   = best_result.pop("test_probs")
    te_labels  = best_result.pop("test_labels")
    val_probs  = best_result.pop("val_probs")
    val_labels = best_result.pop("val_labels")

    model_path    = os.path.join(OUTPUT_DIR, f"lstm_{file_tag}_win{best_w}.pt")
    val_pred_path = os.path.join(OUTPUT_DIR, f"val_predictions_{file_tag}.csv")
    pred_path     = os.path.join(OUTPUT_DIR, f"test_predictions_{file_tag}.csv")
    result_path   = os.path.join(OUTPUT_DIR, f"results_{file_tag}.json")

    torch.save(model.state_dict(), model_path)

    thr = best_result["best_threshold"]

    # val predictions — lấy Date từ val_metadata.parquet của best window
    val_meta_path = os.path.join(best_scenario_dir, "val_metadata.parquet")
    if os.path.exists(val_meta_path):
        val_meta = pd.read_parquet(val_meta_path)
        val_dates = val_meta["Date"].values if "Date" in val_meta.columns else range(len(val_labels))
    else:
        val_dates = range(len(val_labels))
    val_pred_df = pd.DataFrame({
        "Date":        val_dates,
        "y_true":      val_labels,
        "y_prob_lstm": val_probs,
    })
    val_pred_df.to_csv(val_pred_path, index=False)

    pred_df = pd.DataFrame({
        "y_true":      te_labels,
        "y_prob_lstm": te_probs,
        "y_pred_lstm": (te_probs >= thr).astype(int),
    })
    pred_df.to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "file_tag":       file_tag,
            "is_daily":       is_daily,
            "best_window":    best_w,
            "best_val_auc":   float(best_val_auc),
            "best_threshold": float(thr),
            "n_dynamic":      best_result["n_dynamic"],
            "n_static":       best_result["n_static"],
            "cfg":            {k: v for k, v in CFG.items() if k != "device"},
            "splits":         best_result["results"],
        }, f, indent=2)

    print(f"   Saved: {os.path.basename(model_path)}")
    return best_result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {CFG['device']}")
    if CFG["device"] == "cpu":
        print("[INFO] Không có GPU — training sẽ chậm hơn. "
              "Giảm max_epochs hoặc dùng Google Colab nếu cần.")

    all_results: dict = {}

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
        print("  SUMMARY — Best Window, Test Set Performance")
        print(f"{'═'*64}")
        print(f"  {'Tag':<38} {'Win':>4} {'Val AUC':>9} {'Test AUC':>9} {'F1':>7}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            t    = res["results"].get("test", {})
            name = tag.replace("integrated_", "")
            print(
                f"  {name:<38}"
                f" {res['window_size']:>4}"
                f" {res['best_val_auc']:>9.4f}"
                f" {t.get('auc_roc') or 0.0:>9.4f}"
                f" {t.get('f1', 0.0):>7.4f}"
            )
        print(f"\n  Predictions saved to {OUTPUT_DIR}")
        print("  Dùng test_predictions_*.csv + lgbm baseline cho stacking (Tầng 3).")
