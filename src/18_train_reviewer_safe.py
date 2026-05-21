#!/usr/bin/env python3
"""
18_train_reviewer_safe.py

Retrain SeqQC-Former with reviewer-safe options, without modifying the original
pipeline. Supports:
  --feature-mode all_qc
  --feature-mode non_circular_qc
  --feature-mode sequence_only
  --feature-mode qc_only

The non_circular_qc mode removes the QC variables that directly define or
closely mirror the current VAF/normal-alt label rule:
  AD, VAF, normal_alt_fraction, germline_support_flag

Outputs are saved in a new run directory:
  reviewer_safe/runs/<run-name>/
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class H5Dataset(Dataset):
    def __init__(self, h5_path: Path, qc_indices: list[int] | None, sequence_mode: bool = True):
        self.h5_path = str(h5_path)
        self.qc_indices = qc_indices
        self.sequence_mode = sequence_mode
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self.sequences = self._h5["sequences"]
            self.qc_features = self._h5["qc_features"]
            self.labels = self._h5["labels"]

    def __len__(self) -> int:
        self._open()
        return len(self.labels)

    def __getitem__(self, i: int):
        self._open()
        if self.sequence_mode:
            seq = torch.tensor(self.sequences[i], dtype=torch.float32)
            if seq.ndim == 2 and seq.shape[0] != 4:
                seq = seq.transpose(0, 1)  # Lx4 -> 4xL
        else:
            seq = torch.zeros((4, self.sequences.shape[1]), dtype=torch.float32)

        qc = torch.tensor(self.qc_features[i], dtype=torch.float32)
        if self.qc_indices is not None:
            qc = qc[self.qc_indices]
        if qc.numel() == 0:
            qc = torch.zeros(1, dtype=torch.float32)
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return (seq, qc), y, int(i)


class SeqQCNet(nn.Module):
    def __init__(self, qc_dim: int, use_sequence: bool = True, use_qc: bool = True):
        super().__init__()
        self.use_sequence = use_sequence
        self.use_qc = use_qc
        self.seq_out_dim = 64 if use_sequence else 0
        self.qc_out_dim = 16 if use_qc else 0

        if use_sequence:
            self.cnn = nn.Sequential(
                nn.Conv1d(4, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(50)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, batch_first=True, dropout=0.1, activation="gelu"
            )
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if use_qc:
            self.tab = nn.Sequential(
                nn.Linear(qc_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            )

        in_dim = self.seq_out_dim + self.qc_out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_sequence:
            x = self.cnn(seq)
            x = self.pool(x)
            x = x.transpose(1, 2)
            x = self.enc(x)
            parts.append(x.mean(dim=1))
        if self.use_qc:
            parts.append(self.tab(qc))
        z = torch.cat(parts, dim=1)
        return self.fc(z).squeeze(1)


def get_qc_columns(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        names = f.attrs.get("qc_column_names", None)
        if names is None:
            return [f"qc_{i}" for i in range(f["qc_features"].shape[1])]
        return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def choose_features(qc_columns: list[str], mode: str) -> tuple[list[int] | None, bool, bool, list[str]]:
    if mode == "all_qc":
        idx = list(range(len(qc_columns)))
        return idx, True, True, [qc_columns[i] for i in idx]
    if mode == "non_circular_qc":
        excluded = {"AD", "VAF", "normal_alt_fraction", "germline_support_flag"}
        idx = [i for i, c in enumerate(qc_columns) if c not in excluded]
        return idx, True, True, [qc_columns[i] for i in idx]
    if mode == "qc_only":
        idx = list(range(len(qc_columns)))
        return idx, False, True, [qc_columns[i] for i in idx]
    if mode == "sequence_only":
        return [], True, False, []
    raise ValueError(f"Unknown feature mode: {mode}")


def load_split_indices(split_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(split_csv)
    if not {"index", "split"}.issubset(df.columns):
        raise KeyError("Split CSV must contain columns: index, split")
    return {s: df.loc[df["split"] == s, "index"].astype(int).values for s in ["train", "val", "test"]}


def metrics_at_threshold(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def ranking_metrics(y_true, prob) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for (seq, qc), y, idx in loader:
            seq = seq.to(device, non_blocking=True)
            qc = qc.to(device, non_blocking=True)
            logits = model(seq, qc)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            rows.append(pd.DataFrame({
                "index": idx.numpy(),
                "y_true": y.numpy().astype(int),
                "probability": prob,
            }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-h5", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--out-dir", default="reviewer_safe/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--feature-mode", choices=["all_qc", "non_circular_qc", "qc_only", "sequence_only"], default="all_qc")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    features_h5 = Path(args.features_h5)
    split_csv = Path(args.split_csv)
    qc_columns = get_qc_columns(features_h5)
    qc_idx, use_seq, use_qc, used_cols = choose_features(qc_columns, args.feature_mode)

    run_name = args.run_name or f"chrom_{args.feature_mode}_seed{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "feature_mode.json", "w") as f:
        json.dump({
            "feature_mode": args.feature_mode,
            "all_qc_columns": qc_columns,
            "used_qc_columns": used_cols,
            "excluded_qc_columns": [c for c in qc_columns if c not in used_cols],
            "use_sequence": use_seq,
            "use_qc": use_qc,
        }, f, indent=2)

    splits = load_split_indices(split_csv)
    ds = H5Dataset(features_h5, qc_idx, sequence_mode=True)

    with h5py.File(features_h5, "r") as f:
        y_all = f["labels"][:].astype(int)

    train_y = y_all[splits["train"]]
    n_pos = int(train_y.sum())
    n_neg = int(len(train_y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both classes")
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    sample_weights = np.where(train_y == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loaders = {
        "train": DataLoader(Subset(ds, splits["train"]), batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True),
        "val": DataLoader(Subset(ds, splits["val"]), batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(Subset(ds, splits["test"]), batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }

    qc_dim = len(qc_idx) if use_qc else 1
    model = SeqQCNet(qc_dim=qc_dim, use_sequence=use_seq, use_qc=use_qc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_ap = -math.inf
    best_epoch = -1
    wait = 0
    history = []

    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Used QC columns: {used_cols}")
    print(f"Train positives/negatives: {n_pos:,}/{n_neg:,}; pos_weight={pos_weight.item():.3f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (seq, qc), y, _idx in loaders["train"]:
            seq = seq.to(device, non_blocking=True)
            qc = qc.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq, qc)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict(model, loaders["val"], device)
        val_rank = ranking_metrics(val_pred["y_true"].values, val_pred["probability"].values)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val_rank})
        print(f"Epoch {epoch:03d}: loss={np.mean(losses):.4f}, val_AUROC={val_rank['auroc']:.4f}, val_AUPRC={val_rank['auprc']:.4f}")

        if val_rank["auprc"] > best_val_ap:
            best_val_ap = val_rank["auprc"]
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_mode": args.feature_mode,
                "used_qc_columns": used_cols,
                "qc_columns": qc_columns,
                "use_sequence": use_seq,
                "use_qc": use_qc,
                "qc_dim": qc_dim,
                "epoch": epoch,
            }, out_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pred = predict(model, loaders["val"], device)
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    val_thr = pd.DataFrame([metrics_at_threshold(val_pred["y_true"].values, val_pred["probability"].values, t) for t in thresholds])
    best_threshold = float(val_thr.sort_values(["f1", "precision"], ascending=False).iloc[0]["threshold"])

    all_metrics = {"run_name": run_name, "feature_mode": args.feature_mode, "best_epoch": best_epoch, "best_threshold_by_val_f1": best_threshold}
    for split_name in ["val", "test"]:
        pred = val_pred if split_name == "val" else predict(model, loaders[split_name], device)
        pred["predicted"] = (pred["probability"] >= best_threshold).astype(int)
        pred.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
        rank = ranking_metrics(pred["y_true"].values, pred["probability"].values)
        op = metrics_at_threshold(pred["y_true"].values, pred["probability"].values, best_threshold)
        all_metrics[split_name] = {**rank, **op, "n": int(len(pred)), "n_positive": int(pred["y_true"].sum())}

    val_thr.to_csv(out_dir / "validation_threshold_scan.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("✅ Training complete")
    print(f"Run directory: {out_dir.resolve()}")
    print(json.dumps(all_metrics["test"], indent=2))


if __name__ == "__main__":
    main()
