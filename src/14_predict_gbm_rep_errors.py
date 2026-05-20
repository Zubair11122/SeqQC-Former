#!/usr/bin/env python3
"""
Predict replication errors for GBM variants using trained model.

Updated fixes:
1. Uses full HDF5 keys in chrom:pos:ref:alt format.
2. Applies the same training QC scaler from qc_scaler.pkl.
3. Handles duplicate mutation keys correctly.
   - GBM can contain the same key in multiple samples.
   - Therefore, predictions are attached back to gbm_sites.csv by row order,
     not by merge on key.
4. Checks row-order key consistency before saving.
5. Checks checkpoint loading for missing/unexpected keys.
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
out_dir = root / "gbm"

features_h5 = out_dir / "gbm_features.h5"
sites_file = out_dir / "gbm_sites.csv"

ckpt_path = Path(cfg.get("model_ckpt", "rep_error_best_model.ckpt"))
scaler_path = root / "qc_scaler.pkl"

out_pred = out_dir / "gbm_predictions.csv"

threshold = float(cfg.get("gbm_prediction_threshold", 0.75))
predict_batch_size = int(cfg.get("predict_batch_size", 128))


# ----------------------------
# Helpers
# ----------------------------
def require_file(path: Path, label: str):
    """Stop with clear message if file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_features(h5_path: Path):
    """Load GBM HDF5 features."""
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
    """
    Model expects sequences as N x 4 x L.
    Training loader transposed N x L x 4 into N x 4 x L.
    External GBM feature script should already save N x 4 x 501.
    This function safely handles both layouts.
    """
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
    """Load the StandardScaler fitted during training."""
    require_file(path, "Training QC scaler qc_scaler.pkl")

    with open(path, "rb") as f:
        scaler = pickle.load(f)

    if not hasattr(scaler, "transform"):
        raise TypeError(f"Loaded scaler from {path}, but it has no transform() method")

    return scaler


def clean_state_dict_keys(state_dict):
    """Remove common checkpoint prefixes."""
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        for prefix in ["model.", "net.", "_forward_module.", "module."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    return cleaned


def load_model(checkpoint_path: Path, cfg: dict, qc_dim: int, device: torch.device):
    """Load trained model checkpoint safely."""
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
    """Check whether keys look like chrom:pos:ref:alt."""
    if not keys:
        return "empty"

    parts = str(keys[0]).split(":")
    if len(parts) == 4:
        return "chrom:pos:ref:alt"
    if len(parts) == 2:
        return "chrom:pos"
    return f"unknown format: {keys[0]}"


def attach_predictions_by_row_order(sites_df, keys, probs, threshold_value):
    """
    Attach prediction probabilities to gbm_sites.csv by row order.

    This is necessary because the same mutation key can appear in multiple samples.
    A merge on only 'key' is unsafe when keys are duplicated.
    """
    pred_df = pd.DataFrame(
        {
            "key": keys,
            "rep_error_probability": probs,
            "rep_error_predicted": (probs >= threshold_value).astype(int),
        }
    )

    if len(sites_df) != len(pred_df):
        raise ValueError(
            f"Row count mismatch: gbm_sites.csv has {len(sites_df):,} rows, "
            f"but predictions have {len(pred_df):,} rows."
        )

    site_keys = sites_df["key"].astype(str).tolist()
    pred_keys = pred_df["key"].astype(str).tolist()

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
            "Re-run 11_extract_gbm_features.py and make sure it saves "
            "keys = sites['key'].astype(str).tolist()."
        )

    results = sites_df.copy()
    results["rep_error_probability"] = probs
    results["rep_error_predicted"] = (probs >= threshold_value).astype(int)

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 70)
    print("GBM external validation prediction")
    print("=" * 70)

    require_file(features_h5, "GBM features HDF5")
    require_file(sites_file, "GBM sites CSV")
    require_file(ckpt_path, "Model checkpoint")
    require_file(scaler_path, "Training QC scaler")

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

    sites_df = pd.read_csv(sites_file)

    if "key" not in sites_df.columns:
        raise KeyError(f"{sites_file} must contain a 'key' column")

    print(f"  gbm_sites.csv rows: {len(sites_df):,}")
    print(f"  Unique site keys: {sites_df['key'].nunique():,}")
    print(f"  Duplicate key rows: {len(sites_df) - sites_df['key'].nunique():,}")

    if len(sites_df) != len(keys):
        raise ValueError(
            f"gbm_sites.csv row count ({len(sites_df):,}) does not match "
            f"HDF5 key count ({len(keys):,}). Re-run feature extraction."
        )

    h5_key_set = set(map(str, keys))
    site_key_set = set(sites_df["key"].astype(str))
    n_intersect = len(h5_key_set & site_key_set)
    print(
        f"  Unique HDF5 keys matching gbm_sites.csv: "
        f"{n_intersect:,}/{len(site_key_set):,} "
        f"({n_intersect / len(site_key_set) * 100:.2f}%)"
    )

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

    # External GBM MAF-derived QC values can be far outside the training distribution,
    # especially DP and AD. Clip scaled features to avoid extreme out-of-domain inputs.
    qc_clip = float(cfg.get("gbm_qc_scaled_clip", 5.0))
    qc_scaled = np.clip(qc_scaled, -qc_clip, qc_clip).astype(np.float32)

    print(f"  Applied scaled QC clipping: [-{qc_clip}, +{qc_clip}]")
    print(f"  Scaled QC mean/std after clipping: {qc_scaled.mean():.4f}/{qc_scaled.std():.4f}")

    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    qc_dim = qc_scaled.shape[1]
    model = load_model(ckpt_path, cfg, qc_dim, device)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print("\nTesting first batch...")
    test_n = min(5, len(keys))

    with torch.no_grad():
        test_seq = torch.tensor(sequences[:test_n], dtype=torch.float32, device=device)
        test_qc = torch.tensor(qc_scaled[:test_n], dtype=torch.float32, device=device)

        logits = model(test_seq, test_qc)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

    print(f"  First {test_n} probabilities: {np.round(probs, 4)}")

    print(f"\nPredicting {len(keys):,} variants...")
    all_probs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(keys), predict_batch_size), desc="Predicting"):
            end = min(start + predict_batch_size, len(keys))

            seq_batch = torch.tensor(
                sequences[start:end],
                dtype=torch.float32,
                device=device,
            )
            qc_batch = torch.tensor(
                qc_scaled[start:end],
                dtype=torch.float32,
                device=device,
            )

            logits = model(seq_batch, qc_batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.append(probs)

            del seq_batch, qc_batch, logits

    all_probs = np.concatenate(all_probs).astype(np.float32)

    if len(all_probs) != len(keys):
        raise ValueError(
            f"Prediction count mismatch: got {len(all_probs):,}, expected {len(keys):,}"
        )

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

    results["prediction_threshold"] = threshold

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_pred, index=False)

    n_total = len(results)
    n_artifacts = int(results["rep_error_predicted"].sum())
    artifact_rate = n_artifacts / n_total if n_total else 0.0

    print("\n" + "=" * 70)
    print("Prediction summary")
    print("=" * 70)
    print(f"  Total variants: {n_total:,}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Predicted artifacts: {n_artifacts:,}")
    print(f"  Artifact rate: {artifact_rate:.2%}")
    print(f"  Mean probability: {results['rep_error_probability'].mean():.4f}")
    print(f"  Median probability: {results['rep_error_probability'].median():.4f}")
    print(f"  Min probability: {results['rep_error_probability'].min():.4f}")
    print(f"  Max probability: {results['rep_error_probability'].max():.4f}")

    probs = results["rep_error_probability"].values
    print("\nProbability distribution:")
    print(f"  p < 0.25:     {(probs < 0.25).sum():,} ({(probs < 0.25).mean():.1%})")
    print(
        f"  p 0.25-0.50: {((probs >= 0.25) & (probs < 0.50)).sum():,} "
        f"({((probs >= 0.25) & (probs < 0.50)).mean():.1%})"
    )
    print(
        f"  p 0.50-0.75: {((probs >= 0.50) & (probs < 0.75)).sum():,} "
        f"({((probs >= 0.50) & (probs < 0.75)).mean():.1%})"
    )
    print(
        f"  p 0.75-0.90: {((probs >= 0.75) & (probs < 0.90)).sum():,} "
        f"({((probs >= 0.75) & (probs < 0.90)).mean():.1%})"
    )
    print(f"  p >= 0.90:   {(probs >= 0.90).sum():,} ({(probs >= 0.90).mean():.1%})")

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

        sample_out = out_dir / "gbm_sample_statistics.csv"
        sample_stats.to_csv(sample_out, index=False)

        print(f"\nSaved sample statistics: {sample_out}")
        print("\nTop 10 samples by predicted artifact count:")
        for _, row in sample_stats.head(10).iterrows():
            print(
                f"  {row['Tumor_Sample_Barcode']}: "
                f"{int(row['n_predicted_artifacts'])}/{int(row['n_variants'])} "
                f"({row['artifact_rate']:.1%}), "
                f"mean p={row['mean_probability']:.4f}"
            )

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

        gene_out = out_dir / "gbm_gene_statistics.csv"
        gene_stats.to_csv(gene_out, index=False)

        print(f"\nSaved gene statistics: {gene_out}")
        print("\nTop 10 genes by predicted artifact count:")
        for _, row in gene_stats.head(10).iterrows():
            print(
                f"  {row['Hugo_Symbol']}: "
                f"{int(row['n_predicted_artifacts'])}/{int(row['n_variants'])} "
                f"({row['artifact_rate']:.1%}), "
                f"mean p={row['mean_probability']:.4f}"
            )

    print(f"\n✅ Predictions saved to: {out_pred}")
    print("=" * 70)


if __name__ == "__main__":
    main()