#!/usr/bin/env python3
"""
GBM external validation prediction - sensitivity analysis with scaled QC clipping = 3.0

Purpose:
- Does NOT overwrite the main GBM results.
- Reads features from: data_root/gbm/gbm_features.h5
- Reads sites from:    data_root/gbm/gbm_sites.csv
- Writes results to:   data_root/gbm_clip3/

This is intended as a sensitivity analysis for the main GBM prediction run
that used gbm_qc_scaled_clip = 5.0.
"""

from pathlib import Path
import sys
import yaml
import h5py
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make sure src/ can import model_architecture.py
sys.path.insert(0, str(Path(__file__).parent))

from model_architecture import Net


# ----------------------------
# Config
# ----------------------------
CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])

# Input folder remains the main GBM feature folder
input_gbm_dir = root / "gbm"
features_h5 = input_gbm_dir / "gbm_features.h5"
sites_file = input_gbm_dir / "gbm_sites.csv"

# Output folder is separate for sensitivity analysis
out_dir = root / "gbm_clip3"
out_dir.mkdir(parents=True, exist_ok=True)

ckpt_path = Path(cfg.get("model_ckpt", "rep_error_best_model.ckpt"))
scaler_path = root / "qc_scaler.pkl"

out_pred = out_dir / "gbm_predictions_clip3.csv"
out_sample_stats = out_dir / "gbm_sample_statistics_clip3.csv"
out_gene_stats = out_dir / "gbm_gene_statistics_clip3.csv"
out_summary_txt = out_dir / "gbm_clip3_prediction_summary.txt"

threshold = float(cfg.get("gbm_prediction_threshold", 0.75))
predict_batch_size = int(cfg.get("predict_batch_size", 128))

# Fixed sensitivity-analysis clip value
qc_clip = 3.0


# ----------------------------
# Helpers
# ----------------------------
def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_features(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        required = ["sequences", "qc_features", "keys"]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"Missing datasets in {h5_path}: {missing}")

        sequences = f["sequences"][:]
        qc_features = f["qc_features"][:]

        raw_keys = f["keys"][:]
        keys = [
            k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            for k in raw_keys
        ]

        coverage = {}
        for name in ["umap_coverage", "rtim_coverage"]:
            if name in f:
                coverage[name] = f[name][:]

    return sequences, qc_features, keys, coverage


def ensure_sequence_shape(sequences: np.ndarray):
    if sequences.ndim != 3:
        raise ValueError(f"Expected sequences to be 3D, got shape: {sequences.shape}")

    if sequences.shape[1] == 4:
        return sequences.astype(np.float32)

    if sequences.shape[2] == 4:
        print("  Detected N x L x 4 sequence layout; transposing to N x 4 x L")
        return np.transpose(sequences, (0, 2, 1)).astype(np.float32)

    raise ValueError(
        "Could not determine sequence layout. Expected N x 4 x L or N x L x 4, "
        f"got {sequences.shape}"
    )


def load_training_scaler(path: Path):
    require_file(path, "Training QC scaler qc_scaler.pkl")

    with open(path, "rb") as f:
        scaler = pickle.load(f)

    if not hasattr(scaler, "transform"):
        raise TypeError(f"Loaded scaler from {path}, but it has no transform() method")

    return scaler


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["model.", "net.", "_forward_module.", "module."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(checkpoint_path: Path, cfg: dict, qc_dim: int, device: torch.device):
    require_file(checkpoint_path, "Model checkpoint")

    pos_weight = torch.tensor([float(cfg.get("pos_weight", 5.0))], dtype=torch.float32)
    model = Net(cfg, qc_dim, pos_weight)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("\n❌ Missing checkpoint keys:")
        for k in missing[:30]:
            print(f"  - {k}")
        if len(missing) > 30:
            print(f"  ... plus {len(missing) - 30} more")
        raise RuntimeError(
            "Checkpoint did not fully load. Missing model weights found. "
            "Make sure model_architecture.py exactly matches the training model."
        )

    if unexpected:
        print("\n⚠️ Unexpected checkpoint keys:")
        for k in unexpected[:30]:
            print(f"  - {k}")
        if len(unexpected) > 30:
            print(f"  ... plus {len(unexpected) - 30} more")
        print("Continuing because no model weights are missing.")

    return model


def check_key_format(keys):
    if not keys:
        return "empty"

    parts = str(keys[0]).split(":")
    if len(parts) == 4:
        return "chrom:pos:ref:alt"
    if len(parts) == 2:
        return "chrom:pos"
    return f"unknown format: {keys[0]}"


def attach_predictions_by_row_order(sites_df, keys, probs, threshold_value):
    if len(sites_df) != len(keys):
        raise ValueError(
            f"Row count mismatch: gbm_sites.csv has {len(sites_df):,} rows, "
            f"but predictions have {len(keys):,} rows."
        )

    site_keys = sites_df["key"].astype(str).tolist()
    pred_keys = [str(k) for k in keys]

    n_match = sum(a == b for a, b in zip(site_keys, pred_keys))
    match_rate = n_match / len(site_keys) if site_keys else 0.0

    print(
        f"  Row-order key match: {n_match:,}/{len(site_keys):,} "
        f"({match_rate * 100:.2f}%)"
    )

    if n_match != len(site_keys):
        print("\n❌ First mismatched examples:")
        shown = 0
        for i, (site_key, pred_key) in enumerate(zip(site_keys, pred_keys)):
            if site_key != pred_key:
                print(f"  row {i}: gbm_sites.csv={site_key} | HDF5={pred_key}")
                shown += 1
                if shown >= 10:
                    break

        raise ValueError(
            "HDF5 keys are not in the same row order as gbm_sites.csv. "
            "Re-run 11_extract_gbm_features.py."
        )

    results = sites_df.copy()
    results["rep_error_probability"] = probs
    results["rep_error_predicted"] = (probs >= threshold_value).astype(int)
    results["prediction_threshold"] = threshold_value
    results["qc_scaled_clip"] = qc_clip

    return results


def write_summary_text(results, out_path):
    n_total = len(results)
    n_artifacts = int(results["rep_error_predicted"].sum())
    artifact_rate = n_artifacts / n_total if n_total else 0.0
    probs = results["rep_error_probability"].values

    lines = []
    lines.append("=" * 70)
    lines.append("GBM sensitivity analysis: scaled QC clipping = 3.0")
    lines.append("=" * 70)
    lines.append(f"Total variants: {n_total:,}")
    lines.append(f"Threshold: {threshold:.4f}")
    lines.append(f"Scaled QC clip: [-{qc_clip}, +{qc_clip}]")
    lines.append(f"Predicted artifacts: {n_artifacts:,}")
    lines.append(f"Artifact rate: {artifact_rate:.2%}")
    lines.append(f"Mean probability: {results['rep_error_probability'].mean():.4f}")
    lines.append(f"Median probability: {results['rep_error_probability'].median():.4f}")
    lines.append(f"Min probability: {results['rep_error_probability'].min():.4f}")
    lines.append(f"Max probability: {results['rep_error_probability'].max():.4f}")
    lines.append("")
    lines.append("Probability distribution:")
    lines.append(f"p < 0.25:     {(probs < 0.25).sum():,} ({(probs < 0.25).mean():.1%})")
    lines.append(
        f"p 0.25-0.50: {((probs >= 0.25) & (probs < 0.50)).sum():,} "
        f"({((probs >= 0.25) & (probs < 0.50)).mean():.1%})"
    )
    lines.append(
        f"p 0.50-0.75: {((probs >= 0.50) & (probs < 0.75)).sum():,} "
        f"({((probs >= 0.50) & (probs < 0.75)).mean():.1%})"
    )
    lines.append(
        f"p 0.75-0.90: {((probs >= 0.75) & (probs < 0.90)).sum():,} "
        f"({((probs >= 0.75) & (probs < 0.90)).mean():.1%})"
    )
    lines.append(f"p >= 0.90:   {(probs >= 0.90).sum():,} ({(probs >= 0.90).mean():.1%})")

    out_path.write_text("\n".join(lines))
    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 70)
    print("GBM external validation prediction - sensitivity analysis")
    print("Scaled QC clipping fixed at 3.0")
    print("=" * 70)

    require_file(features_h5, "GBM features HDF5")
    require_file(sites_file, "GBM sites CSV")
    require_file(ckpt_path, "Model checkpoint")
    require_file(scaler_path, "Training QC scaler")

    print("\nInput/output:")
    print(f"  Input features: {features_h5}")
    print(f"  Input sites:    {sites_file}")
    print(f"  Output folder:  {out_dir}")

    # Load features
    print("\nLoading GBM features...")
    sequences, qc_raw, keys, coverage = load_features(features_h5)
    sequences = ensure_sequence_shape(sequences)

    print(f"  Variants: {len(keys):,}")
    print(f"  Sequences: {sequences.shape}")
    print(f"  QC raw: {qc_raw.shape}")
    print(f"  Key format: {check_key_format(keys)}")

    if len(keys) != sequences.shape[0] or len(keys) != qc_raw.shape[0]:
        raise ValueError(
            "Feature length mismatch: "
            f"keys={len(keys):,}, sequences={sequences.shape[0]:,}, qc={qc_raw.shape[0]:,}"
        )

    # Load sites
    sites_df = pd.read_csv(sites_file)

    if "key" not in sites_df.columns:
        raise KeyError(f"{sites_file} must contain a 'key' column")

    print(f"  gbm_sites.csv rows: {len(sites_df):,}")
    print(f"  Unique site keys: {sites_df['key'].nunique():,}")
    print(f"  Duplicate key rows: {len(sites_df) - sites_df['key'].nunique():,}")

    if len(sites_df) != len(keys):
        raise ValueError(
            f"gbm_sites.csv row count ({len(sites_df):,}) does not match "
            f"HDF5 key count ({len(keys):,})."
        )

    h5_key_set = set(map(str, keys))
    site_key_set = set(sites_df["key"].astype(str))
    n_intersect = len(h5_key_set & site_key_set)
    print(
        f"  Unique HDF5 keys matching gbm_sites.csv: "
        f"{n_intersect:,}/{len(site_key_set):,} "
        f"({n_intersect / len(site_key_set) * 100:.2f}%)"
    )

    # Scale and clip QC
    print("\nLoading training QC scaler...")
    scaler = load_training_scaler(scaler_path)

    expected_dim = getattr(scaler, "n_features_in_", qc_raw.shape[1])
    if qc_raw.shape[1] != expected_dim:
        raise ValueError(
            f"QC feature dimension mismatch: GBM has {qc_raw.shape[1]}, "
            f"but scaler expects {expected_dim}."
        )

    qc_raw = qc_raw.astype(np.float32)
    qc_scaled = scaler.transform(qc_raw).astype(np.float32)

    print(f"  Raw QC mean/std: {qc_raw.mean():.4f}/{qc_raw.std():.4f}")
    print(f"  Scaled QC mean/std before clipping: {qc_scaled.mean():.4f}/{qc_scaled.std():.4f}")

    qc_scaled = np.clip(qc_scaled, -qc_clip, qc_clip).astype(np.float32)

    print(f"  Applied scaled QC clipping: [-{qc_clip}, +{qc_clip}]")
    print(f"  Scaled QC mean/std after clipping: {qc_scaled.mean():.4f}/{qc_scaled.std():.4f}")

    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    qc_dim = qc_scaled.shape[1]
    model = load_model(ckpt_path, cfg, qc_dim, device)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Test first batch
    print("\nTesting first batch...")
    test_n = min(5, len(keys))

    with torch.no_grad():
        test_seq = torch.tensor(sequences[:test_n], dtype=torch.float32, device=device)
        test_qc = torch.tensor(qc_scaled[:test_n], dtype=torch.float32, device=device)

        logits = model(test_seq, test_qc)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

    print(f"  First {test_n} probabilities: {np.round(probs, 4)}")

    # Predict all
    print(f"\nPredicting {len(keys):,} variants...")
    all_probs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(keys), predict_batch_size), desc="Predicting"):
            end = min(start + predict_batch_size, len(keys))

            seq_batch = torch.tensor(sequences[start:end], dtype=torch.float32, device=device)
            qc_batch = torch.tensor(qc_scaled[start:end], dtype=torch.float32, device=device)

            logits = model(seq_batch, qc_batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.append(probs)

            del seq_batch, qc_batch, logits

    all_probs = np.concatenate(all_probs).astype(np.float32)

    if len(all_probs) != len(keys):
        raise ValueError(
            f"Prediction count mismatch: got {len(all_probs):,}, expected {len(keys):,}"
        )

    # Build results by row order
    print("\nBuilding prediction table...")
    results = attach_predictions_by_row_order(
        sites_df=sites_df,
        keys=keys,
        probs=all_probs,
        threshold_value=threshold,
    )

    for name, arr in coverage.items():
        if len(arr) != len(results):
            raise ValueError(
                f"Coverage feature {name} has length {len(arr):,}, "
                f"but results have {len(results):,} rows."
            )
        results[name] = arr

    results.to_csv(out_pred, index=False)

    # Sample statistics
    if "Tumor_Sample_Barcode" in results.columns:
        sample_stats = (
            results.groupby("Tumor_Sample_Barcode")
            .agg(
                n_variants=("key", "count"),
                n_predicted_artifacts=("rep_error_predicted", "sum"),
                artifact_rate=("rep_error_predicted", "mean"),
                mean_probability=("rep_error_probability", "mean"),
            )
            .reset_index()
            .sort_values(["n_predicted_artifacts", "artifact_rate"], ascending=False)
        )
        sample_stats.to_csv(out_sample_stats, index=False)
    else:
        sample_stats = None

    # Gene statistics
    if "Hugo_Symbol" in results.columns:
        gene_stats = (
            results.groupby("Hugo_Symbol")
            .agg(
                n_variants=("key", "count"),
                n_predicted_artifacts=("rep_error_predicted", "sum"),
                artifact_rate=("rep_error_predicted", "mean"),
                mean_probability=("rep_error_probability", "mean"),
            )
            .reset_index()
            .sort_values(["n_predicted_artifacts", "artifact_rate"], ascending=False)
        )
        gene_stats.to_csv(out_gene_stats, index=False)
    else:
        gene_stats = None

    # Summary
    print("\n" + "=" * 70)
    summary_text = write_summary_text(results, out_summary_txt)
    print(summary_text)

    print("\nSaved files:")
    print(f"  Predictions:       {out_pred}")
    print(f"  Summary text:      {out_summary_txt}")
    if sample_stats is not None:
        print(f"  Sample statistics: {out_sample_stats}")
    if gene_stats is not None:
        print(f"  Gene statistics:   {out_gene_stats}")

    print("\n✅ Clip 3.0 sensitivity analysis completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
