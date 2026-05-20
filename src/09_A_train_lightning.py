#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import yaml
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import gc

# ---------------- Set random seed for reproducibility ----------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------- Config with defaults ----------------
cfg = yaml.safe_load(open("config.yaml"))

# Set default values if not present in config
cfg.setdefault('window_bp', cfg.get('sequence_window', 250))
cfg.setdefault('pos_weight', 5.0)
cfg.setdefault('gradient_clip_val', 1.0)
cfg.setdefault('accumulate_grad_batches', 2)
cfg.setdefault('precision', '16-mixed')
cfg.setdefault('warmup_epochs', 5)
cfg.setdefault('weight_decay', 0.01)
cfg.setdefault('early_stop_patience', 15)

# Convert to appropriate types
cfg['batch_size'] = int(cfg['batch_size'])
cfg['epochs'] = int(cfg['epochs'])
cfg['sequence_window'] = int(cfg['sequence_window'])
cfg['window_bp'] = int(cfg['window_bp'])
cfg['learning_rate'] = float(cfg['learning_rate'])
cfg['weight_decay'] = float(cfg['weight_decay'])
cfg['pos_weight'] = float(cfg['pos_weight'])
cfg['gradient_clip_val'] = float(cfg['gradient_clip_val'])
cfg['early_stop_patience'] = int(cfg['early_stop_patience'])
cfg['warmup_epochs'] = int(cfg['warmup_epochs'])

# Print config for verification
print("="*60)
print("Configuration:")
print("="*60)
for key, value in cfg.items():
    if not any(sensitive in key.lower() for sensitive in ['bam', 'password', 'key']):
        print(f"  {key}: {value}")
print("="*60)

root = Path(cfg["data_root"])
h5_path = root / "features.h5"
splits_path = root / "splits.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

# Set precision for better performance
torch.set_float32_matmul_precision('medium')

# ---------------- Dataset (CORRECTED for your HDF5 structure) ----------------
class H5Set(Dataset):
    def __init__(self, path: Path, augment=False):
        self.path = str(path)
        self._h5 = None
        self.augment = augment

    def _ensure_open(self):
        if self._h5 is None:
            # Open in read-only mode with libver='latest' for better performance
            self._h5 = h5py.File(self.path, "r", libver='latest', swmr=True)
            # CORRECTED: Use your actual dataset names
            self.x_seq = self._h5["sequences"]      # Changed from "seq"
            self.x_qc = self._h5["qc_features"]    # Changed from "qc"
            self.y_ds = self._h5["labels"]         # Changed from "y"

    def __len__(self):
        self._ensure_open()
        return len(self.y_ds)

    def __getitem__(self, i):
        self._ensure_open()
        seq = torch.tensor(self.x_seq[i], dtype=torch.float32)
        # Ensure correct shape (channels, length)
        if seq.ndim == 2 and seq.shape[0] != 4:
            seq = seq.transpose(0, 1)
        qc = torch.tensor(self.x_qc[i], dtype=torch.float32)
        y = torch.tensor(int(self.y_ds[i]), dtype=torch.float32)
        
        # Augmentation for training (reverse complement)
        if self.augment and np.random.rand() < 0.5:  # Increased to 50% for better augmentation
            seq = seq.flip(1)  # Reverse sequence along length dimension
        
        return (seq, qc), y

# ---------------- Load splits ----------------
print("\nLoading data splits...")
splits = pd.read_csv(splits_path)
train_idx = splits[splits["split"] == "train"]["index"].values
val_idx = splits[splits["split"] == "val"]["index"].values
test_idx = splits[splits["split"] == "test"]["index"].values

print(f"Train samples: {len(train_idx):,}")
print(f"Validation samples: {len(val_idx):,}")
print(f"Test samples: {len(test_idx):,}")

# CORRECTED: Create separate dataset instances for train and validation
# This prevents the augmentation flag from affecting both datasets
train_full_ds = H5Set(h5_path, augment=True)
val_full_ds = H5Set(h5_path, augment=False)

train_ds = torch.utils.data.Subset(train_full_ds, train_idx)
val_ds = torch.utils.data.Subset(val_full_ds, val_idx)

# Calculate class weights efficiently (without loading all data into memory)
print("\nCalculating class weights...")
# Efficient counting using numpy on a subset
positive_count = 0
batch_size_count = 10000
for i in range(0, len(train_idx), batch_size_count):
    batch_indices = train_idx[i:i+batch_size_count]
    y_batch = np.array([train_full_ds[j][1].item() for j in batch_indices])
    positive_count += np.sum(y_batch == 1)

negative_count = len(train_idx) - positive_count
counts = [negative_count, positive_count]

print(f"Training set - Negatives: {counts[0]:,}, Positives: {counts[1]:,}")
print(f"Imbalance ratio: {counts[0]/max(counts[1], 1):.2f}:1")

# Use pos_weight from config or calculate from data
if 'pos_weight' in cfg and cfg['pos_weight'] > 0:
    pos_weight = torch.tensor([cfg['pos_weight']], dtype=torch.float32)
    print(f"Using configured positive weight: {pos_weight.item():.3f}")
else:
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32)
    print(f"Calculated positive weight: {pos_weight.item():.3f}")

# Calculate sample weights efficiently
sample_weights = np.zeros(len(train_idx))
for i, idx in enumerate(train_idx):
    y_val = train_full_ds[idx][1].item()
    sample_weights[i] = 1.0 / counts[int(y_val)]

# Normalize weights
sample_weights = sample_weights / sample_weights.sum()
sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

# Free memory
del sample_weights
gc.collect()

# Dataloaders with optimized settings
batch_size = min(int(cfg["batch_size"]), 32)  # REDUCED from 64 to 32 for memory safety
num_workers = 2  # REDUCED from 4 to avoid memory overhead

print(f"\nUsing batch size: {batch_size}")
print(f"Using num_workers: {num_workers}")

train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    sampler=sampler, 
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False,
    prefetch_factor=2 if num_workers > 0 else None
)

val_loader = DataLoader(
    val_ds, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2 if num_workers > 0 else None
)

# ---------------- Model (with memory optimizations) ----------------
class Net(pl.LightningModule):
    def __init__(self, cfg, qc_dim, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        
        self.seq_len = 2 * int(cfg.get("window_bp", cfg.get("sequence_window", 250))) + 1
        self.qc_dim = qc_dim
        
        # CNN with reduced channels for memory efficiency
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),  # Reduced from 64
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),  # Reduced from 128
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),  # Reduced from 128
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(50)
        
        # Transformer encoder with smaller model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,  # Reduced from 128
            nhead=4, 
            batch_first=True, 
            dropout=0.1,
            activation='gelu'
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Reduced from 3
        
        # QC feature processing
        self.tab = nn.Sequential(
            nn.Linear(qc_dim, 32),  # Reduced from 64
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(32, 16),  # Reduced from 32
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        # Final classifier (reduced size)
        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 64),  # Reduced from 128
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),  # Reduced from 64
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Loss and metrics
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        self.train_auroc = tm.AUROC(task="binary")
        self.val_auroc = tm.AUROC(task="binary")
        self.val_ap = tm.AveragePrecision(task="binary")
        
    def forward(self, seq, qc):
        # Sequence pathway
        x = self.cnn(seq)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.enc(x)
        x = x.mean(dim=1)  # Global mean pooling
        
        # QC pathway
        q = self.tab(qc)
        
        # Combine
        combined = torch.cat([x, q], dim=1)
        return self.fc(combined).squeeze(1)
    
    def training_step(self, batch, batch_idx):
        (seq, qc), y = batch
        logits = self(seq, qc)
        loss = self.bce(logits, y)
        
        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, y.int())
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_auroc", self.train_auroc, prog_bar=True, on_epoch=True, on_step=False)
        
        # Clear cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        (seq, qc), y = batch
        logits = self(seq, qc)
        loss = self.bce(logits, y)
        
        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y.int())
        self.val_ap.update(probs, y.int())
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        val_ap = self.val_ap.compute()
        self.log("val_auroc", val_auroc, prog_bar=True, on_epoch=True)
        self.log("val_ap", val_ap, prog_bar=True, on_epoch=True)
        self.val_auroc.reset()
        self.val_ap.reset()
        
        # Clear cache after validation
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        # Use weight_decay from config
        weight_decay = self.hparams['cfg'].get('weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams['cfg']['learning_rate'],
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

# ---------------- Get QC dimension (CORRECTED) ----------------
with h5py.File(h5_path, "r") as f:
    qc_dim = f["qc_features"].shape[1]  # Changed from "qc"
    print(f"\nQC feature dimension: {qc_dim}")

# ---------------- Initialize Model ----------------
net = Net(cfg, qc_dim, pos_weight)

# Log model size
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# ---------------- Configure Trainer ----------------
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best-epoch={epoch:02d}-auroc={val_auroc:.4f}",
    monitor="val_auroc",
    mode="max",
    save_top_k=2,  # Reduced from 3 to save disk space
    save_last=True
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="val_auroc",
    mode="max",
    patience=cfg['early_stop_patience'],
    verbose=True
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# Mixed precision
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    precision = "bf16-mixed"
    print("Using BF16 mixed precision")
else:
    precision = cfg.get('precision', '16-mixed')
    print(f"Using {precision} precision")

# Use gradient checkpointing for memory efficiency
if torch.cuda.is_available():
    torch.set_autocast_enabled(True)

trainer = pl.Trainer(
    max_epochs=cfg["epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None,
    precision=precision if torch.cuda.is_available() else "32-true",
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    log_every_n_steps=10,
    gradient_clip_val=cfg['gradient_clip_val'],
    accumulate_grad_batches=cfg['accumulate_grad_batches'],
    enable_progress_bar=True,
    deterministic=False,
    # Memory optimization flags
    enable_checkpointing=True,
    default_root_dir=str(checkpoint_dir),
    num_sanity_val_steps=0  # Disable sanity validation to save memory
)

# ---------------- Train ----------------
print("\n" + "="*60)
print("Starting Training...")
print("="*60)
print(f"Max epochs: {cfg['epochs']}")
print(f"Batch size: {batch_size}")
print(f"Effective batch size: {batch_size * cfg['accumulate_grad_batches']}")
print(f"Learning rate: {cfg['learning_rate']}")
print(f"Weight decay: {cfg['weight_decay']}")
print(f"Gradient clip: {cfg['gradient_clip_val']}")
print(f"Accumulate grad batches: {cfg['accumulate_grad_batches']}")
print(f"Early stop patience: {cfg['early_stop_patience']}")
print("="*60)

last_checkpoint = checkpoint_dir / "last.ckpt"
try:
    trainer.fit(
        net, 
        train_loader, 
        val_loader,
        ckpt_path=str(last_checkpoint) if last_checkpoint.exists() else None
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("\n❌ Out of Memory Error!")
        print("Suggestions:")
        print("  1. Reduce batch_size further in config.yaml")
        print("  2. Reduce num_workers to 0 or 1")
        print("  3. Use gradient accumulation (already enabled)")
        print("  4. Disable pin_memory in DataLoader")
    raise e

# ---------------- Final Evaluation ----------------
print("\n" + "="*60)
print("Final Evaluation on Validation Set")
print("="*60)

net.eval()
if torch.cuda.is_available():
    net.to(device)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch_idx, ((seq, qc), y) in enumerate(val_loader):
        if torch.cuda.is_available():
            seq = seq.to(device, non_blocking=True)
            qc = qc.to(device, non_blocking=True)
        
        logits = net(seq, qc)
        probs = torch.sigmoid(logits)
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())
        
        # Clear batch from GPU
        if torch.cuda.is_available():
            del seq, qc, logits, probs
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels)

auroc = roc_auc_score(all_labels, all_probs)
auprc = average_precision_score(all_labels, all_probs)

print(f"\n✅ Final Validation Results:")
print(f"   AUROC: {auroc:.4f}")
print(f"   AUPRC: {auprc:.4f}")

# Calculate best threshold
best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.3, 0.8, 0.05):
    preds = (all_probs > threshold).astype(int)
    f1 = f1_score(all_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n   Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

# Classification report
preds = (all_probs > best_threshold).astype(int)
print(f"\nClassification Report (threshold={best_threshold:.2f}):")
print(classification_report(all_labels, preds, target_names=['Negative', 'Positive']))

# ---------------- Save Model ----------------
final_model_path = Path("rep_error_best_model.ckpt")
trainer.save_checkpoint(str(final_model_path))
print(f"\n✅ Model saved to: {final_model_path}")

# Also save the best threshold
with open("best_threshold.txt", "w") as f:
    f.write(f"{best_threshold}\n")
    f.write(f"F1: {best_f1}\n")
    f.write(f"AUROC: {auroc}\n")
    f.write(f"AUPRC: {auprc}\n")
print(f"✅ Best threshold saved to: best_threshold.txt")

# ---------------- Test set evaluation ----------------
print("\n" + "="*60)
print("Evaluating on Test Set")
print("="*60)

if len(test_idx) > 0:
    test_full_ds = H5Set(h5_path, augment=False)
    test_ds = torch.utils.data.Subset(test_full_ds, test_idx)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    all_probs_test = []
    all_labels_test = []
    
    with torch.no_grad():
        for batch_idx, ((seq, qc), y) in enumerate(test_loader):
            if torch.cuda.is_available():
                seq = seq.to(device, non_blocking=True)
                qc = qc.to(device, non_blocking=True)
            
            logits = net(seq, qc)
            probs = torch.sigmoid(logits)
            
            all_probs_test.extend(probs.cpu().numpy())
            all_labels_test.extend(y.numpy())
            
            # Clear batch from GPU
            if torch.cuda.is_available():
                del seq, qc, logits, probs
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
    
    all_probs_test = np.array(all_probs_test).flatten()
    all_labels_test = np.array(all_labels_test)
    
    test_auroc = roc_auc_score(all_labels_test, all_probs_test)
    test_auprc = average_precision_score(all_labels_test, all_probs_test)
    
    print(f"\n✅ Test Set Results:")
    print(f"   AUROC: {test_auroc:.4f}")
    print(f"   AUPRC: {test_auprc:.4f}")
    
    # Test set classification report
    test_preds = (all_probs_test > best_threshold).astype(int)
    print(f"\nTest Set Classification Report (threshold={best_threshold:.2f}):")
    print(classification_report(all_labels_test, test_preds, target_names=['Negative', 'Positive']))
else:
    print("No test set found (test_idx is empty)")

print("\n" + "="*60)
print("🎉 Training completed successfully!")
print("="*60)

# Optional: Plot ROC curve
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f'Validation (AUC = {auroc:.4f})')
    
    if len(test_idx) > 0:
        fpr_test, tpr_test, _ = roc_curve(all_labels_test, all_probs_test)
        plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auroc:.4f})', linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    print("\n✅ ROC curve saved to: roc_curves.png")
except Exception as e:
    print(f"\nCould not plot ROC curve: {e}")

# Final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()