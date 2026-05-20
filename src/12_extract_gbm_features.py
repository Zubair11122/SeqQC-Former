#!/usr/bin/env python3
"""
11_extract_gbm_features_updated.py

Extract external GBM features in the SAME semantic format expected by the
trained replication-error model.

Key fixes vs the previous version:
  1. HDF5 keys are saved as full variant keys: chrom:pos:ref:alt
     so prediction output can merge back to gbm_sites.csv correctly.
  2. AD is stored as raw tumor alternate allele count, matching the
     training QC feature meaning better than storing a fraction.
  3. VAF is stored separately as t_alt_count / t_depth.
  4. N bases are encoded as all-zero, matching the training one-hot encoder.
  5. QC feature column names/order are saved as HDF5 metadata.

Expected input:
  data_root/gbm/gbm_sites.csv from 10_prepare_gbm_maf_sites.py

Expected output:
  data_root/gbm/gbm_features.h5
"""

from pathlib import Path
import sys
import yaml
import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

try:
    import pyBigWig
except ImportError:
    pyBigWig = None


CFG = Path("config.yaml")
QC_COLUMNS = [
    "DP",
    "AD",
    "VAF",
    "MQ",
    "SB",
    "tumor_strand_bias",
    "tumor_orientation_bias",
    "tumor_clipped_fraction",
    "tumor_mismatch_fraction",
    "normal_alt_fraction",
    "germline_support_flag",
]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def first_existing_numeric(row: pd.Series, names, default=np.nan) -> float:
    """Return first valid numeric value from possible column names."""
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                pass
    return default


def resolve_chrom_key(fasta: Fasta, chrom) -> str | None:
    """Find chromosome name in FASTA, trying with and without chr prefix."""
    c = str(chrom).strip()
    candidates = []

    if c.startswith("chr"):
        candidates.extend([c, c.replace("chr", "", 1)])
    else:
        candidates.extend([f"chr{c}", c])

    if c in {"MT", "M", "chrM", "chrMT"}:
        candidates.extend(["chrM", "MT", "M", "chrMT"])

    for candidate in candidates:
        if candidate in fasta:
            return candidate
    return None


def one_hot_encode_training_style(seq: str, max_len: int) -> np.ndarray:
    """
    One-hot encode as (4, L), matching what the model receives.

    Training one-hot behavior:
      A/C/G/T get one-hot values.
      N or any unknown base remains all-zero.
    """
    bases = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, max_len), dtype=np.float32)

    seq_upper = str(seq).upper()[:max_len]
    for i, base in enumerate(seq_upper):
        idx = bases.get(base)
        if idx is not None:
            encoded[idx, i] = 1.0

    return encoded


def extract_sequences(sites: pd.DataFrame, fasta: Fasta, window_bp: int) -> tuple[np.ndarray, int]:
    seq_len = 2 * window_bp + 1
    sequences = np.zeros((len(sites), 4, seq_len), dtype=np.float32)
    failed = 0

    print("\nExtracting sequence context...")
    for i, row in tqdm(sites.iterrows(), total=len(sites), desc="Sequences"):
        chrom = row["Chromosome"]
        pos = int(row["Start_Position"])
        chrom_key = resolve_chrom_key(fasta, chrom)

        if chrom_key is None:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: chromosome not found in FASTA: {chrom}")
            continue

        # MAF Start_Position is 1-based. pyfaidx slicing is 0-based half-open.
        start = pos - window_bp - 1
        end = pos + window_bp

        # Handle edge-of-chromosome sites by padding N on the left/right.
        left_pad = max(0, -start)
        start = max(0, start)

        try:
            seq = str(fasta[chrom_key][start:end]).upper()
            if left_pad:
                seq = "N" * left_pad + seq
            if len(seq) < seq_len:
                seq = seq + "N" * (seq_len - len(seq))
            elif len(seq) > seq_len:
                seq = seq[:seq_len]

            if len(seq) != seq_len:
                failed += 1
                continue

            sequences[i] = one_hot_encode_training_style(seq, seq_len)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: failed at {chrom}:{pos}: {e}")

    return sequences, failed


def build_qc_features(sites: pd.DataFrame) -> np.ndarray:
    """
    Build 11 QC features in the same order as training.

    Training order:
      DP, AD, VAF, MQ, SB,
      tumor_strand_bias, tumor_orientation_bias,
      tumor_clipped_fraction, tumor_mismatch_fraction,
      normal_alt_fraction, germline_support_flag

    For MAF-only GBM data, some read-level features are unavailable, so they
    are filled with conservative defaults. If your MAF has equivalent columns,
    this function will use them where possible.
    """
    qc = np.zeros((len(sites), len(QC_COLUMNS)), dtype=np.float32)

    print("\nBuilding QC features...")
    for i, row in tqdm(sites.iterrows(), total=len(sites), desc="QC features"):
        dp = first_existing_numeric(row, ["DP", "t_depth", "tumor_depth"], default=np.nan)
        ad = first_existing_numeric(row, ["AD", "t_alt_count", "tumor_alt_count"], default=np.nan)
        ref_count = first_existing_numeric(row, ["t_ref_count", "tumor_ref_count"], default=np.nan)

        # If depth is absent but ref/alt counts exist, reconstruct depth.
        if not np.isfinite(dp):
            if np.isfinite(ref_count) and np.isfinite(ad):
                dp = ref_count + ad
            else:
                dp = 0.0

        # If AD is absent, use 0 rather than a fake fraction.
        if not np.isfinite(ad):
            ad = 0.0

        # VAF: prefer explicit VAF columns; otherwise compute from AD/DP.
        vaf = first_existing_numeric(
            row,
            ["VAF", "vaf", "tumor_vaf", "t_vaf", "i_TumorVAF_WU"],
            default=np.nan,
        )
        if not np.isfinite(vaf):
            vaf = ad / dp if dp > 0 else 0.0
        # If VAF appears as percent, convert to fraction.
        if vaf > 1.0 and vaf <= 100.0:
            vaf = vaf / 100.0

        normal_alt = first_existing_numeric(
            row,
            ["normal_alt_count", "n_alt_count", "Normal_Alt_Count"],
            default=0.0,
        )
        normal_dp = first_existing_numeric(
            row,
            ["normal_depth", "n_depth", "Normal_Depth"],
            default=np.nan,
        )
        normal_alt_fraction = normal_alt / normal_dp if np.isfinite(normal_dp) and normal_dp > 0 else 0.0

        qc[i, 0] = dp                                      # DP: depth
        qc[i, 1] = ad                                      # AD: raw alt count, not fraction
        qc[i, 2] = vaf                                     # VAF: fraction
        qc[i, 3] = first_existing_numeric(row, ["MQ", "MappingQuality", "mapping_quality"], default=60.0)
        qc[i, 4] = first_existing_numeric(row, ["SB", "strand_bias"], default=0.0)
        qc[i, 5] = first_existing_numeric(row, ["tumor_strand_bias"], default=0.0)
        qc[i, 6] = first_existing_numeric(row, ["tumor_orientation_bias"], default=0.0)
        qc[i, 7] = first_existing_numeric(row, ["tumor_clipped_fraction"], default=0.0)
        qc[i, 8] = first_existing_numeric(row, ["tumor_mismatch_fraction"], default=0.0)
        qc[i, 9] = normal_alt_fraction
        qc[i, 10] = 1.0 if normal_alt > 0 else 0.0          # germline_support_flag approximation

    qc = np.nan_to_num(qc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return qc


def extract_bigwig_at_sites(sites: pd.DataFrame, bw_path: str | Path, name: str) -> np.ndarray | None:
    if pyBigWig is None:
        print(f"  pyBigWig not installed; skipping {name}")
        return None

    bw_path = Path(bw_path)
    if not bw_path.exists():
        print(f"  {name} bigWig not found: {bw_path}")
        return None

    print(f"  Extracting {name} from {bw_path}")
    values = np.zeros(len(sites), dtype=np.float32)

    try:
        bw = pyBigWig.open(str(bw_path))
        chroms = bw.chroms()
        for i, row in tqdm(sites.iterrows(), total=len(sites), desc=name):
            chrom = str(row["Chromosome"])
            pos = int(row["Start_Position"])

            candidates = [chrom, f"chr{chrom}" if not chrom.startswith("chr") else chrom.replace("chr", "", 1)]
            if chrom in {"MT", "M"}:
                candidates.extend(["chrM", "M", "MT"])

            chrom_key = next((c for c in candidates if c in chroms), None)
            if chrom_key is None:
                continue

            try:
                vals = bw.values(chrom_key, max(0, pos - 1), pos)
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                values[i] = float(np.mean(vals)) if vals else 0.0
            except Exception:
                values[i] = 0.0
        bw.close()
        return values
    except Exception as e:
        print(f"  WARNING: could not read {name}: {e}")
        return None


def main():
    cfg = load_config(CFG)
    root = Path(cfg["data_root"])
    out_dir = root / "gbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    window_bp = int(cfg.get("window_bp", cfg.get("sequence_window", 250)))
    seq_len = 2 * window_bp + 1

    sites_file = out_dir / "gbm_sites.csv"
    out_h5 = out_dir / "gbm_features.h5"

    if not sites_file.exists():
        raise FileNotFoundError(f"Missing {sites_file}. Run 10_prepare_gbm_maf_sites.py first.")
    if "reference_fasta" not in cfg:
        raise KeyError("config.yaml must contain reference_fasta")

    sites = pd.read_csv(sites_file)
    required = ["key", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]
    missing = [c for c in required if c not in sites.columns]
    if missing:
        raise ValueError(f"gbm_sites.csv is missing required columns: {missing}")

    print("=" * 70)
    print("GBM feature extraction - training-matched format")
    print("=" * 70)
    print(f"Sites: {len(sites):,}")
    print(f"Window: {window_bp} bp each side; sequence length: {seq_len}")

    print("\nLoading reference FASTA...")
    fasta = Fasta(str(cfg["reference_fasta"]), rebuild=False)
    print(f"Loaded {len(fasta.keys())} FASTA records")

    sequences, failed_sites = extract_sequences(sites, fasta, window_bp)
    qc_features = build_qc_features(sites)

    coverage_features = {}
    print("\nOptional coverage features...")
    for cfg_key, h5_name in [("umap_bw", "umap_coverage"), ("rtim_bw", "rtim_coverage")]:
        bw_path = cfg.get(cfg_key)
        if bw_path:
            arr = extract_bigwig_at_sites(sites, bw_path, h5_name)
            if arr is not None:
                coverage_features[h5_name] = arr.astype(np.float32)
                print(f"  Saved optional feature: {h5_name}")

    print("\nWriting HDF5...")
    full_keys = sites["key"].astype(str).tolist()
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip", chunks=True)
        f.create_dataset("qc_features", data=qc_features, compression="gzip", chunks=True)
        f.create_dataset("labels", data=np.full(len(sites), -1, dtype=np.int32), compression="gzip")
        f.create_dataset("keys", data=np.array(full_keys, dtype="S"), compression="gzip")

        for name, arr in coverage_features.items():
            f.create_dataset(name, data=arr, compression="gzip")

        f.attrs["n_samples"] = len(sites)
        f.attrs["seq_length"] = seq_len
        f.attrs["sequence_shape"] = "N,4,L"
        f.attrs["n_qc_features"] = len(QC_COLUMNS)
        f.attrs["qc_column_names"] = np.array(QC_COLUMNS, dtype="S")
        f.attrs["window_bp"] = window_bp
        f.attrs["key_format"] = "chrom:pos:ref:alt"
        f.attrs["notes"] = "AD is raw alt count; VAF is fraction; N bases are all-zero."

    file_size = out_h5.stat().st_size / 1e9
    print("\n✅ GBM features saved")
    print(f"  Output: {out_h5}")
    print(f"  Sequences: {sequences.shape}")
    print(f"  QC features: {qc_features.shape}")
    print(f"  Failed sequence extraction: {failed_sites:,}/{len(sites):,} ({failed_sites / max(len(sites), 1):.2%})")
    print(f"  HDF5 key example: {full_keys[0] if full_keys else 'NA'}")
    print(f"  File size: {file_size:.2f} GB")

    if failed_sites / max(len(sites), 1) > 0.01:
        print("\n⚠️ More than 1% of sites failed sequence extraction. Check genome build and chr naming.")


if __name__ == "__main__":
    main()
