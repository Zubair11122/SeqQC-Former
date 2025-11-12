# 03_A_train_lightning.py — safe, GPU, resume-ready (fixed labels loading)

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # safer on mounted disks

import yaml, torch, h5py, numpy as np, torch.nn as nn, pytorch_lightning as pl
import torchmetrics as tm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------- Config -----------------
CFG_PATH = Path(__file__).with_name("config.yaml")
cfg = yaml.safe_load(open(CFG_PATH))
cfg['batch_size']    = int(cfg['batch_size'])
cfg['epochs']        = int(cfg['epochs'])
cfg['pos_weight']    = float(cfg['pos_weight'])
cfg['learning_rate'] = float(cfg['learning_rate'])
cfg['window_bp']     = int(cfg['window_bp'])

data_root = Path(cfg['data_root']).expanduser()
if not data_root.is_absolute():
    data_root = (Path(__file__).parent / data_root).resolve()
h5_path = data_root / "features.h5"

print(f"\n[INFO] config:   {CFG_PATH.resolve()}")
print(f"[INFO] data_root:{data_root}")
print(f"[INFO] h5 file:  {h5_path}")
if not h5_path.exists():
    raise FileNotFoundError(f"features.h5 not found at {h5_path}")

# conservative loader settings to avoid HDF5 lockups; you can increase later
SAFE_BATCH   = min(cfg['batch_size'], 64)
NUM_WORKERS  = 0
PIN_MEMORY   = False
PERSISTENT   = False

# ----------------- Dataset (per-worker handle) -----------------
class H5Set(Dataset):
    """Each worker opens its own HDF5 handle (prevents deadlocks)."""
    def __init__(self, path: Path):
        self.path = str(path)
        self._h5 = None
        self.length = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
            self.x_seq = self._h5["seq"]  # (N, 4, L) or (N, L, 4)
            self.x_qc  = self._h5["qc"]   # (N, 7)
            self.y_ds  = self._h5["y"]    # dataset; not a NumPy array
            self.length = len(self.y_ds)

    def __len__(self):
        if self.length is not None:
            return self.length
        with h5py.File(self.path, "r") as f:
            return len(f["y"])

    def __getitem__(self, i):
        self._ensure_open()
        seq = torch.tensor(self.x_seq[i], dtype=torch.float32)
        if seq.ndim == 2 and seq.shape[0] != 4 and seq.shape[1] == 4:
            seq = seq.transpose(0, 1)  # (L,4)->(4,L)
        qc  = torch.tensor(self.x_qc[i], dtype=torch.float32)
        y   = torch.tensor(int(self.y_ds[i]))
        return (seq, qc), y

# ----------------- Load data & split -----------------
print("\n[INFO] Loading dataset...")
full = H5Set(h5_path)
n_total = len(full)
train_size = int(0.8 * n_total)
val_size   = n_total - train_size
train_data, val_data = torch.utils.data.random_split(
    full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Read labels once from disk (works even before dataset is opened)
with h5py.File(h5_path, "r") as f:
    y_all = np.asarray(f["y"], dtype=int)

# Weighted sampler for imbalance (train only)
train_idx = np.array(train_data.indices, dtype=int)
train_labels = y_all[train_idx]
classes = max(2, int(train_labels.max()) + 1) if train_labels.size else 2
counts = np.bincount(train_labels, minlength=classes).astype(np.float64)
counts[counts == 0] = 1.0
weights = 1.0 / counts[train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

# ----------------- Model -----------------
class Net(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cnn = nn.Sequential(
            nn.Conv1d(4,   64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128,128, 3, padding=1), nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.tab = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32))
        self.fc  = nn.Sequential(nn.Linear(160, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

        self.register_buffer("pos_weight", torch.tensor([cfg['pos_weight']], dtype=torch.float32))
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.auroc = torch.jit.ignore(lambda: tm.AUROC(task="binary"))() if False else tm.AUROC(task="binary")
        self.ap    = tm.AveragePrecision(task="binary")

    def forward(self, seq, qc):
        x = self.cnn(seq).transpose(1, 2)  # (B,L,128)
        x = self.enc(x)[:, 0, :]
        q = self.tab(qc)
        return self.fc(torch.cat([x, q], 1)).squeeze(1)

    def training_step(self, batch, _):
        (seq, qc), y = batch
        logits = self(seq, qc)
        loss = self.bce(logits, y.float())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        (seq, qc), y = batch
        logits = self(seq, qc)
        p = torch.sigmoid(logits)
        self.auroc.update(p, y.int()); self.ap.update(p, y.int())
        self.log("val_loss", self.bce(logits, y.float()), prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.auroc.compute(), prog_bar=True)
        self.log("val_ap",    self.ap.compute(),    prog_bar=True)
        self.auroc.reset(); self.ap.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams['learning_rate'])
        sched = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5),
                 "monitor": "val_auroc"}
        return {"optimizer": opt, "lr_scheduler": sched}

# ----------------- Dataloaders -----------------
train_loader = DataLoader(
    train_data, batch_size=SAFE_BATCH, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT
)
val_loader = DataLoader(
    val_data, batch_size=SAFE_BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT
)

# ----------------- Trainer (GPU + checkpoints) -----------------
precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
ckpt_dir = data_root / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath=str(ckpt_dir), filename="epoch{epoch:02d}-auroc{val_auroc:.4f}",
    monitor="val_auroc", mode="max", save_top_k=1, save_last=True
)
earlystop_cb = pl.callbacks.EarlyStopping(monitor="val_auroc", mode="max", patience=10)

trainer = pl.Trainer(
    max_epochs=cfg['epochs'],
    accelerator="gpu", devices=1,
    precision=precision, log_every_n_steps=20,
    enable_progress_bar=True,
    callbacks=[checkpoint_cb, earlystop_cb],
)

print("\n[INFO] Starting training… (auto-resume if last.ckpt exists)")
net = Net(cfg)
last_ckpt = ckpt_dir / "last.ckpt"
trainer.fit(net, train_loader, val_loader, ckpt_path=str(last_ckpt) if last_ckpt.exists() else None)

final_ckpt = data_root / "rep_error_net.ckpt"
trainer.save_checkpoint(str(final_ckpt))
print(f"[INFO] Done. Final checkpoint: {final_ckpt}")
