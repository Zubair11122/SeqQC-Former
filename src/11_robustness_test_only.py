#!/usr/bin/env python3
"""
09_robustness_test_only_fixed.py

Evaluate robustness of the trained sequence+QC model on the held-out test split by adding Gaussian noise to scaled QC features.

This version is matched to the current project pipeline:
  - features.h5 datasets: sequences, qc_features, labels
  - splits.csv columns: index, split
  - training checkpoint: Lightning checkpoint from 07_A_train_lightning.py
  - model architecture: same CNN + Transformer + QC branch as Net in 07_A_train_lightning.py
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import glob
import yaml
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path.resolve()}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "data_root" not in cfg:
        raise KeyError("config.yaml must contain data_root")
    return cfg


def resolve_checkpoint(root: Path, cfg: dict) -> Path:
    """Find the most likely checkpoint produced by 07_A_train_lightning.py."""
    candidates = []

    # Explicit config option, if you add one later
    for key in ["checkpoint_path", "ckpt_path", "model_checkpoint"]:
        if key in cfg:
            candidates.append(Path(cfg[key]))

    # Final checkpoint saved by 07_A_train_lightning.py
    candidates += [
        Path("rep_error_best_model.ckpt"),
        root / "rep_error_best_model.ckpt",
        Path("checkpoints") / "last.ckpt",
        root / "checkpoints" / "last.ckpt",
    ]

    # Best validation checkpoint saved by Lightning ModelCheckpoint
    candidates += [Path(p) for p in glob.glob("checkpoints/best-*.ckpt")]
    candidates += [Path(p) for p in glob.glob(str(root / "checkpoints" / "best-*.ckpt"))]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        msg = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Could not find a checkpoint. Checked:\n"
            f"{msg}\n\n"
            "Expected the checkpoint from 07_A_train_lightning.py, usually "
            "rep_error_best_model.ckpt or checkpoints/best-*.ckpt."
        )

    # Prefer best checkpoints over last, otherwise first candidate order
    best = [p for p in existing if "best-" in p.name]
    return sorted(best)[-1] if best else existing[0]


def load_threshold(root: Path, default=0.5) -> float:
    """Use the threshold saved by training if available; otherwise use 0.5."""
    candidates = [Path("best_threshold.txt"), root / "best_threshold.txt"]
    for path in candidates:
        if path.exists():
            first = path.read_text().strip().splitlines()[0]
            try:
                return float(first)
            except ValueError:
                pass
    return float(default)


class EvalNet(nn.Module):
    """
    Architecture matched to Net in 07_A_train_lightning.py, without Lightning.
    """

    def __init__(self, cfg, qc_dim):
        super().__init__()

        self.seq_len = 2 * int(cfg.get("window_bp", cfg.get("sequence_window", 250))) + 1
        self.qc_dim = int(qc_dim)

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

        combined = torch.cat([x, q], dim=1)
        return self.fc(combined).squeeze(1)


def extract_state_dict(ckpt):
    """Support Lightning checkpoints and plain torch state_dict checkpoints."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    clean = {}
    for k, v in state.items():
        # Common Lightning/wrapper prefixes
        for prefix in ["model.", "net.", "module."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    return clean


def filter_state_dict_for_model(state, model):
    """Keep only parameters/buffers that exist in the evaluation model with matching shapes.

    PyTorch Lightning checkpoints may include training-only buffers such as
    bce.pos_weight. That key is useful during training but is not part of this
    lightweight EvalNet, so it should be ignored at evaluation time.
    """
    expected = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state.items():
        if k in expected and tuple(v.shape) == tuple(expected[k].shape):
            filtered[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"Skipping {len(skipped)} checkpoint key(s) not used by EvalNet:")
        for k in skipped[:20]:
            print(f"  - {k}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    missing = [k for k in expected.keys() if k not in filtered]
    if missing:
        raise RuntimeError(
            "Checkpoint is missing required EvalNet key(s): "
            + ", ".join(missing[:20])
            + (f" ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        )

    return filtered


def binary_auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_auprc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    total_pos = (y_true == 1).sum()
    if total_pos == 0:
        return np.nan

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos

    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def compute_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    accuracy = float((tp + tn) / max(len(y_true), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

    return {
        "AUROC": binary_auroc(y_true, probs),
        "AUPRC": binary_auprc(y_true, probs),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }



def main():
    cfg = load_config()
    root = Path(cfg["data_root"])

    features_file = root / "features.h5"
    splits_file = root / "splits.csv"
    out_file = root / "robustness_test_only.csv"

    if not features_file.exists():
        raise FileNotFoundError(f"Missing {features_file}")
    if not splits_file.exists():
        raise FileNotFoundError(f"Missing {splits_file}")

    checkpoint_file = resolve_checkpoint(root, cfg)
    threshold = load_threshold(root, default=0.5)

    print("Features:", features_file)
    print("Splits:", splits_file)
    print("Checkpoint:", checkpoint_file)
    print("Threshold:", threshold)

    splits = pd.read_csv(splits_file)
    if not {"index", "split"}.issubset(splits.columns):
        raise ValueError("splits.csv must contain columns: index, split")

    test_idx = splits.loc[splits["split"] == "test", "index"].to_numpy(dtype=np.int64)
    test_idx = np.sort(test_idx)

    if len(test_idx) == 0:
        raise ValueError("No test samples found in splits.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    with h5py.File(features_file, "r") as f:
        required = ["sequences", "qc_features", "labels"]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"features.h5 is missing datasets: {missing}. Found: {list(f.keys())}")

        qc_dim = f["qc_features"].shape[1]
        n_total = f["labels"].shape[0]

    if test_idx.max() >= n_total:
        raise IndexError("splits.csv contains index values beyond labels length in features.h5")

    ckpt = torch.load(checkpoint_file, map_location=device)
    model = EvalNet(cfg, qc_dim=qc_dim).to(device)
    state = extract_state_dict(ckpt)
    state = filter_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    batch_size = int(cfg.get("eval_batch_size", min(int(cfg.get("batch_size", 32)), 256)))
    noise_levels = cfg.get("robustness_noise_levels", [0.0, 0.01, 0.05, 0.10, 0.15, 0.20])
    noise_levels = [float(x) for x in noise_levels]

    rng = np.random.default_rng(42)
    results = []

    for sigma in noise_levels:
        all_probs = []
        all_labels = []

        with h5py.File(features_file, "r") as f, torch.no_grad():
            seq_ds = f["sequences"]
            qc_ds = f["qc_features"]
            y_ds = f["labels"]

            for start in range(0, len(test_idx), batch_size):
                idx = test_idx[start:start + batch_size]

                seq = seq_ds[idx].astype(np.float32)  # expected N,L,4
                qc = qc_ds[idx].astype(np.float32)
                y = y_ds[idx].astype(np.int32)

                if sigma > 0:
                    qc = qc + rng.normal(0, sigma, qc.shape).astype(np.float32)

                if seq.shape[1] != 4 and seq.shape[2] == 4:
                    seq = np.transpose(seq, (0, 2, 1))  # N,4,L

                seq_t = torch.from_numpy(seq).to(device)
                qc_t = torch.from_numpy(qc).to(device)

                logits = model(seq_t, qc_t)
                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

                all_probs.append(probs)
                all_labels.append(y)

        probs = np.concatenate(all_probs)
        y_test = np.concatenate(all_labels).astype(int)

        metrics_dict = compute_metrics(y_test, probs, threshold)
        row = {
            "noise_sigma": sigma,
            "n": len(y_test),
            "positives": int((y_test == 1).sum()),
            "negatives": int((y_test == 0).sum()),
            "threshold": threshold,
            "checkpoint": str(checkpoint_file),
            "mean_probability": float(probs.mean()),
            **metrics_dict,
        }

        results.append(row)
        print(row)

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
