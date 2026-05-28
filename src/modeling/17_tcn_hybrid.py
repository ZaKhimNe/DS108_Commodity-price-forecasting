"""
tcn_hybrid.py — Tầng 2: TCN Hybrid (Dynamic 3D + Static 2D)
============================================================
Temporal Convolutional Network với dilated causal convolutions.
Load tensors từ data/tensors/win_{W}/{file_tag}/ (output tensor_packing.py).

Ưu điểm so với LSTM:
- Receptive field (RF) rõ ràng và có thể tính toán trước
- Không có hidden state → train nhanh hơn ~2-3×, fully parallelizable
- Không bị vanishing gradient (residual connections xuyên suốt)
- Thường thắng LSTM trên financial data ngắn vì local pattern quan trọng hơn

Công thức RF: RF = 1 + (kernel_size - 1) × sum(2^i for i in range(n_blocks))
  4 blocks, k=3: RF = 1 + 2×(1+2+4+8) = 31
  5 blocks, k=3: RF = 1 + 2×(1+2+4+8+16) = 63  ← daily window=45 cần cái này

Chạy:
    python src/modeling/tcn_hybrid.py
"""

import os
import json
import warnings
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "17_tcn_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = dict(
    # Model
    n_filters     = 64,     # Số filters mỗi TCN block (giữ đồng nhất qua tất cả blocks)
    kernel_size   = 3,      # Kernel size — k=3 là chuẩn cho financial, k=5 nếu muốn RF lớn hơn
    n_blocks      = 4,      # Số TCN blocks — tự động thêm 1 block nếu RF < window_size
    head_hidden   = 64,     # Dense head hidden size
    dropout       = 0.20,   # TCN ít cần dropout hơn LSTM vì residual connections ổn định hơn

    # Training
    lr            = 5e-4,   # TCN thường cần LR thấp hơn LSTM một chút
    weight_decay  = 1e-4,
    max_epochs    = 120,
    batch_daily   = 64,
    batch_weekly  = 32,
    patience      = 15,
    lr_patience   = 5,
    clip_norm     = 1.0,
    max_spw       = 10.0,

    # Misc
    seed          = 42,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)

FILES_DAILY  = ["integrated_coffee_daily",  "integrated_corn_daily"]
FILES_WEEKLY = ["integrated_coffee_weekly", "integrated_corn_weekly"]

# v5: expand to {14,30} — TCN now covers both short and medium windows
# LSTM expands to {14,30,45} — both models still overlap less than before P3
ALLOWED_WINDOWS_DAILY  = {14, 30}
ALLOWED_WINDOWS_WEEKLY = None   # weekly has few windows — no restriction


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, X_dyn: np.ndarray, X_stat: np.ndarray, y: np.ndarray):
        # TCN nhận (N, D, W) — channel-first, khác với LSTM (N, W, D)
        # Transpose ngay khi load để tránh nhầm lẫn sau này
        self.X_dyn  = torch.FloatTensor(X_dyn).permute(0, 2, 1)   # (N, D, W)
        self.X_stat = torch.FloatTensor(X_stat) \
                      if X_stat.ndim == 2 and X_stat.shape[1] > 0 \
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


# ─── TCN Building Blocks ──────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal Convolution 1D — chỉ nhìn vào quá khứ (không look-ahead).

    Kỹ thuật: pad (kernel_size - 1) × dilation bên trái, cắt bên phải.
    Điều này đảm bảo output[t] chỉ phụ thuộc vào input[t-k:t], không có
    thông tin tương lai nào bị rò vào.

    Input shape:  (N, C_in, W)
    Output shape: (N, C_out, W)  — W không thay đổi (same length)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Cắt bỏ phần padding bên phải để giữ W không đổi
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """
    Một TCN block gồm 2 lớp Causal Conv1D với cùng dilation.

    Cấu trúc:
        x ──► CausalConv1D ──► BN ──► ReLU ──► Dropout
              CausalConv1D ──► BN ──► ReLU ──► Dropout ──► out
        │                                                    │
        └──► [1×1 Conv nếu cần reshape channels] ──────────►+── output

    Residual connection:
    - Nếu in_channels == out_channels: x + out (trực tiếp)
    - Nếu khác nhau: dùng 1×1 Conv để project x về đúng dimension

    Tại sao 2 conv layers per block?
    - 1 layer: RF tăng tuyến tính với dilation
    - 2 layers: RF tăng nhanh hơn và feature mixing tốt hơn
    - Đây là convention từ paper gốc "An Empirical Evaluation of Generic
      Convolutional and Recurrent Networks for Sequence Modeling" (Bai 2018)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        # Residual projection nếu channels khác nhau
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop2(out)

        return self.relu(out + residual)


# ─── TCN Hybrid Model ─────────────────────────────────────────────────────────

class TCNHybrid(nn.Module):
    """
    Architecture:
        Dynamic (N, D, W) ──► TCNBlock(d=1)
                           ──► TCNBlock(d=2)
                           ──► TCNBlock(d=4)
                           ──► TCNBlock(d=8)   [+ thêm d=16 nếu cần RF lớn hơn]
                           ──► last timestep [:, :, -1]
                                     │
        Static (N, S) ──────────────►Concat
                                     │
                                 BN + Dense(64) + Dropout
                                     │
                                 Dense(1) [logits]

    Dilations: [1, 2, 4, 8, ...] — mỗi block tăng gấp đôi → RF tăng theo cấp số nhân.

    Output: LOGITS — dùng torch.sigmoid() khi inference.
    """

    def __init__(
        self,
        n_dynamic:   int,
        n_static:    int,
        n_filters:   int   = 64,
        kernel_size: int   = 3,
        n_blocks:    int   = 4,
        head_hidden: int   = 64,
        dropout:     float = 0.2,
        window_size: int   = 14,    # để auto-adjust n_blocks nếu RF < window
    ):
        super().__init__()

        # Auto-adjust n_blocks để RF >= window_size
        n_blocks = self._auto_n_blocks(n_blocks, kernel_size, window_size)
        self.n_blocks  = n_blocks
        self.rf        = self._compute_rf(kernel_size, n_blocks)

        # Xây dựng stack TCN blocks với dilation tăng dần
        blocks = []
        for i in range(n_blocks):
            in_ch  = n_dynamic if i == 0 else n_filters
            out_ch = n_filters
            dil    = 2 ** i
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dil, dropout))

        self.tcn_stack = nn.Sequential(*blocks)

        # Classification head
        head_in = n_filters + n_static
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),  # logits
        )

        self._init_weights()

    @staticmethod
    def _compute_rf(kernel_size: int, n_blocks: int) -> int:
        """Tính receptive field: RF = 1 + (k-1) * sum(2^i)"""
        return 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))

    @staticmethod
    def _auto_n_blocks(n_blocks: int, kernel_size: int, window_size: int) -> int:
        """Tăng n_blocks cho đến khi RF >= window_size."""
        rf = 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))
        while rf < window_size and n_blocks < 8:
            n_blocks += 1
            rf = 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_blocks))
        return n_blocks

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_dyn:  torch.Tensor,   # (N, D, W) — channel-first
        x_stat: torch.Tensor,   # (N, S)
    ) -> torch.Tensor:           # (N,) logits

        # TCN stack — mỗi block giữ nguyên (N, n_filters, W)
        feat = self.tcn_stack(x_dyn)        # (N, n_filters, W)

        # Chỉ lấy last timestep — tương đương h_T trong LSTM
        last = feat[:, :, -1]               # (N, n_filters)

        # Concat với static features
        combined = torch.cat([last, x_stat], dim=1) \
                   if x_stat.shape[1] > 0 \
                   else last

        return self.head(combined).squeeze(1)   # (N,) logits


# ─── Training Utilities ───────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_auc   = -np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_auc: float, model: nn.Module) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc   = val_auc
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


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
    all_probs, all_labels = [], []
    total_loss = 0.0
    for x_dyn, x_stat, y in loader:
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        logits = model(x_dyn, x_stat)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    probs  = np.array(all_probs,  dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return total_loss / len(loader.dataset), auc, probs, labels


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def compute_full_metrics(name, y_true, y_prob, threshold) -> dict:
    y_pred   = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    auc_roc  = float(roc_auc_score(y_true, y_prob))           if has_both else float("nan")
    pr_auc   = float(average_precision_score(y_true, y_prob)) if has_both else float("nan")
    f1       = float(f1_score(y_true, y_pred, zero_division=0))
    prec     = float(precision_score(y_true, y_pred, zero_division=0))
    base     = float(y_true.mean())
    lift     = (prec / base) if base > 0 else float("nan")
    print(
        f"   [{name:<6}]  n={len(y_true):>4}  base={base:.3f}  thr={threshold:.2f}"
        f"  AUC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}  Lift={lift:.2f}x"
    )
    return dict(
        split=name, n_samples=int(len(y_true)), n_positive=int(y_true.sum()),
        base_rate=base, threshold=threshold,
        auc_roc=auc_roc if not math.isnan(auc_roc) else None,
        pr_auc=pr_auc   if not math.isnan(pr_auc)  else None,
        f1=f1, precision=prec,
        lift=lift if not math.isnan(lift) else None,
    )


# ─── Tensor Loading ───────────────────────────────────────────────────────────

def load_tensors(scenario_dir: str) -> dict | None:
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


# ─── Experiment Runner ────────────────────────────────────────────────────────

def run_window_experiment(
    file_tag:    str,
    window_size: int,
    tensors:     dict,
    batch_size:  int,
    device:      torch.device,
    cfg:         dict,
) -> dict | None:
    set_seed(cfg["seed"])

    X_tr_dyn, X_vl_dyn, X_te_dyn = tensors["X_train_dyn"], tensors["X_val_dyn"], tensors["X_test_dyn"]
    X_tr_st,  X_vl_st,  X_te_st  = tensors["X_train_stat"], tensors["X_val_stat"], tensors["X_test_stat"]
    y_tr,     y_vl,     y_te      = tensors["y_train"], tensors["y_val"], tensors["y_test"]

    if len(y_tr) == 0 or len(y_vl) == 0:
        print("   [SKIP] Tensor rỗng.")
        return None

    n_dynamic = X_tr_dyn.shape[2]
    n_static  = X_tr_st.shape[1] if X_tr_st.ndim == 2 else 0
    pos       = y_tr.sum()
    neg       = len(y_tr) - pos
    spw       = float(min(neg / pos, cfg["max_spw"])) if pos > 0 else 1.0

    model = TCNHybrid(
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

    print(
        f"   Window={window_size}  RF={rf}  blocks={n_blocks}"
        f"  dyn={n_dynamic}  static={n_static}"
        f"  n_train={len(y_tr)}  base={y_tr.mean():.3f}"
        f"  spw={spw:.2f}  params={n_params:,}"
    )

    if rf < window_size:
        print(f"   [WARN] RF={rf} < window={window_size} — model không phủ hết lịch sử!")
    elif rf > window_size * 2:
        print(f"   [INFO] RF={rf} >> window={window_size} — có thể giảm n_blocks để nhẹ hơn.")

    train_loader = make_loader(X_tr_dyn, X_tr_st, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X_vl_dyn, X_vl_st, y_vl, batch_size)
    test_loader  = make_loader(X_te_dyn, X_te_st, y_te, batch_size)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=cfg["lr_patience"], min_lr=1e-5,
    )
    early_stop = EarlyStopping(patience=cfg["patience"])

    print(f"   {'Epoch':>5}  {'TrLoss':>9}  {'VaLoss':>9}  {'Val AUC':>9}  {'LR':>9}")
    for epoch in range(1, cfg["max_epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg["clip_norm"])
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
    _, _, vl_p, vl_l = eval_epoch(model, val_loader,   criterion, device)
    _, _, tr_p, tr_l = eval_epoch(model, train_loader,  criterion, device)
    _, _, te_p, te_l = eval_epoch(model, test_loader,   criterion, device)

    best_thr = find_optimal_threshold(vl_l, vl_p)
    results  = {
        "train": compute_full_metrics("TRAIN", tr_l, tr_p, best_thr),
        "val":   compute_full_metrics("VAL",   vl_l, vl_p, best_thr),
        "test":  compute_full_metrics("TEST",  te_l, te_p, best_thr),
    }

    gap = (results["train"]["auc_roc"] or 0) - (results["val"]["auc_roc"] or 0)
    if gap > 0.12:
        print(f"   [WARN] Overfit gap={gap:.4f}: tăng dropout hoặc giảm n_filters.")

    return dict(
        window_size    = window_size,
        n_blocks       = n_blocks,
        rf             = rf,
        best_val_auc   = early_stop.best_auc,
        best_threshold = best_thr,
        n_dynamic      = n_dynamic,
        n_static       = n_static,
        n_params       = n_params,
        results        = results,
        model          = model,
        test_probs     = te_p,
        test_labels    = te_l,
        val_probs      = vl_p,
        val_labels     = vl_l,
    )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(file_tag: str, is_daily: bool) -> dict | None:
    print(f"\n{'═'*64}")
    print(f"  {file_tag.upper()}")
    print(f"{'═'*64}")

    batch_size = CFG["batch_daily"] if is_daily else CFG["batch_weekly"]
    device     = torch.device(CFG["device"])

    available_windows = []
    allowed = ALLOWED_WINDOWS_DAILY if is_daily else ALLOWED_WINDOWS_WEEKLY
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
        print(f"   [SKIP] Không tìm thấy tensors cho {file_tag}")
        return None

    print(f"   Window sizes: {[w for w, _ in available_windows]}")

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

    best_w = best_result["window_size"]
    print(f"\n  ── Best window: {best_w}  RF={best_result['rf']}"
          f"  val AUC={best_val_auc:.4f} ──")

    model      = best_result.pop("model")
    te_probs   = best_result.pop("test_probs")
    te_labels  = best_result.pop("test_labels")
    val_probs  = best_result.pop("val_probs")
    val_labels = best_result.pop("val_labels")
    thr        = best_result["best_threshold"]

    model_path    = os.path.join(OUTPUT_DIR, f"tcn_{file_tag}_win{best_w}.pt")
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
        "Date":       val_dates,
        "y_true":     val_labels,
        "y_prob_tcn": val_probs,
    }).to_csv(val_pred_path, index=False)

    pd.DataFrame({
        "y_true":     te_labels,
        "y_prob_tcn": te_probs,
        "y_pred_tcn": (te_probs >= thr).astype(int),
    }).to_csv(pred_path, index=False)

    with open(result_path, "w") as f:
        json.dump({
            "file_tag":       file_tag,
            "is_daily":       is_daily,
            "best_window":    best_w,
            "best_val_auc":   float(best_val_auc),
            "best_threshold": float(thr),
            "n_blocks":       best_result["n_blocks"],
            "rf":             best_result["rf"],
            "n_params":       best_result["n_params"],
            "cfg":            {k: v for k, v in CFG.items() if k != "device"},
            "splits":         best_result["results"],
        }, f, indent=2)

    print(f"   Saved: {os.path.basename(model_path)}")
    return best_result


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {CFG['device']}")
    if CFG["device"] == "cpu":
        print("[INFO] CPU mode — TCN train nhanh hơn LSTM trên CPU vì fully parallelizable.")

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
        print("  SUMMARY — TCN Best Results")
        print(f"{'═'*64}")
        print(f"  {'Tag':<34} {'Win':>4} {'RF':>5} {'Blks':>5}"
              f" {'Val AUC':>9} {'Test AUC':>9} {'F1':>7}")
        print(f"  {'-'*62}")
        for tag, res in all_results.items():
            t    = res["results"].get("test", {})
            name = tag.replace("integrated_", "")
            print(
                f"  {name:<34}"
                f" {res['window_size']:>4}"
                f" {res['rf']:>5}"
                f" {res['n_blocks']:>5}"
                f" {res['best_val_auc']:>9.4f}"
                f" {t.get('auc_roc') or 0.0:>9.4f}"
                f" {t.get('f1', 0.0):>7.4f}"
            )
        print(f"\n  Dùng test_predictions_*.csv (cột y_prob_tcn) cho stacking (Tầng 3).")
        print(f"  Kết hợp với lgbm/y_prob_lgb + lstm/y_prob_lstm → meta-learner.")
