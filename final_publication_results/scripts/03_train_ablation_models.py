import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("final_publication_results")
TABLES = ROOT / "tables"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

H5 = ROOT / "features.h5"
SPLITS = TABLES / "splits.csv"

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
PATIENCE = 5
THRESHOLD = 0.75

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        self.h5 = None

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.seq = self.h5["sequences"]
            self.qc = self.h5["qc_features"]
            self.y = self.h5["labels"]

    def __len__(self):
        self._open()
        return len(self.y)

    def __getitem__(self, idx):
        self._open()
        seq = torch.tensor(self.seq[idx], dtype=torch.float32)
        if seq.ndim == 2 and seq.shape[0] != 4:
            seq = seq.transpose(0, 1)
        qc = torch.tensor(self.qc[idx], dtype=torch.float32)
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        return seq, qc, y

class QCOnly(nn.Module):
    def __init__(self, qc_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(qc_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq, qc):
        return self.net(qc).squeeze(1)

class SequenceOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(50)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq, qc):
        x = self.cnn(seq)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)

class SeqQCFormer(nn.Module):
    def __init__(self, qc_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(50)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.tab = nn.Sequential(
            nn.Linear(qc_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq, qc):
        x = self.cnn(seq)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.enc(x)
        x = x.mean(dim=1)
        q = self.tab(qc)
        return self.fc(torch.cat([x, q], dim=1)).squeeze(1)

def evaluate(model, loader):
    model.eval()
    y_all = []
    p_all = []

    with torch.no_grad():
        for seq, qc, y in loader:
            seq = seq.to(DEVICE)
            qc = qc.to(DEVICE)
            logits = model(seq, qc)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_all.append(y.numpy())
            p_all.append(probs)

    y_true = np.concatenate(y_all).astype(int)
    y_prob = np.concatenate(p_all)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }

def train_one(name, model, train_loader, val_loader, test_loader, pos_weight):
    print(f"\n===== Training {name} =====")
    model = model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_auroc = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for seq, qc, y in train_loader:
            seq = seq.to(DEVICE)
            qc = qc.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq, qc)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader)
        print(
            f"{name} epoch {epoch:02d} "
            f"loss={np.mean(losses):.6f} "
            f"val_AUROC={val_metrics['AUROC']:.6f} "
            f"val_AUPRC={val_metrics['AUPRC']:.6f}"
        )

        if val_metrics["AUROC"] > best_val_auroc:
            best_val_auroc = val_metrics["AUROC"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"Early stopping {name} at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS / f"{name}.pt")

    test_metrics = evaluate(model, test_loader)
    test_metrics["Model"] = name
    test_metrics["best_val_AUROC"] = best_val_auroc
    return test_metrics

def main():
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not H5.exists():
        raise FileNotFoundError(H5)
    if not SPLITS.exists():
        raise FileNotFoundError(SPLITS)

    ds = H5Dataset(H5)
    splits = pd.read_csv(SPLITS)

    train_idx = splits.loc[splits["split"] == "train", "index"].to_numpy()
    val_idx = splits.loc[splits["split"] == "val", "index"].to_numpy()
    test_idx = splits.loc[splits["split"] == "test", "index"].to_numpy()

    with h5py.File(H5, "r") as f:
        y_train = f["labels"][train_idx]
        qc_dim = f["qc_features"].shape[1]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    print("Train positives:", n_pos)
    print("Train negatives:", n_neg)
    print("pos_weight:", pos_weight)
    print("qc_dim:", qc_dim)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results = []
    results.append(train_one("QC_only", QCOnly(qc_dim), train_loader, val_loader, test_loader, pos_weight))
    results.append(train_one("Sequence_only", SequenceOnly(), train_loader, val_loader, test_loader, pos_weight))
    results.append(train_one("SeqQC_Former_retrained", SeqQCFormer(qc_dim), train_loader, val_loader, test_loader, pos_weight))

    out = TABLES / "ablation_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)

    print("\nAblation results:")
    print(df.to_string(index=False))
    print("Saved:", out)

if __name__ == "__main__":
    main()