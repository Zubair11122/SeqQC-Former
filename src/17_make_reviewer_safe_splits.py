#!/usr/bin/env python3
"""
17_make_reviewer_safe_splits.py

Create reviewer-safe chromosome-held-out splits for SeqQC-Former without
modifying the original random split files.

Default split:
  train: chromosomes 1-16
  val:   chromosomes 17-18
  test:  chromosomes 19-22

Inputs:
  - features.h5 with labels dataset
  - qc_readlevel.csv with key column formatted as chrom:pos:ref:alt

Outputs:
  - reviewer_safe/reviewer_metadata.csv
  - reviewer_safe/splits_chrom_reviewer_safe.csv
  - reviewer_safe/chromosome_split_summary.csv
  - reviewer_safe/split_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def norm_chrom(x: str) -> str:
    c = str(x).replace("chr", "").strip()
    if c.upper() in {"M", "MT"}:
        return "MT"
    return c.upper() if c.upper() in {"X", "Y"} else c


def parse_chrom_list(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item and item.replace("-", "").replace(" ", "").isdigit():
            a, b = item.split("-", 1)
            out.update(str(i) for i in range(int(a), int(b) + 1))
        else:
            out.add(norm_chrom(item))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-h5", required=True, help="Path to features.h5")
    p.add_argument("--qc-csv", required=True, help="Path to qc_readlevel.csv")
    p.add_argument("--out-dir", default="reviewer_safe", help="Output directory")
    p.add_argument("--train-chroms", default="1-16")
    p.add_argument("--val-chroms", default="17-18")
    p.add_argument("--test-chroms", default="19-22")
    args = p.parse_args()

    features_h5 = Path(args.features_h5)
    qc_csv = Path(args.qc_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_h5.exists():
        raise FileNotFoundError(features_h5)
    if not qc_csv.exists():
        raise FileNotFoundError(qc_csv)

    with h5py.File(features_h5, "r") as f:
        if "labels" not in f:
            raise KeyError("features.h5 must contain dataset 'labels'")
        labels = f["labels"][:].astype(int)

    df = pd.read_csv(qc_csv)
    if "key" not in df.columns:
        raise KeyError("qc_readlevel.csv must contain a 'key' column in chrom:pos:ref:alt format")
    if len(df) != len(labels):
        raise ValueError(f"Row mismatch: qc CSV has {len(df):,}; HDF5 labels have {len(labels):,}")

    key_parts = df["key"].astype(str).str.split(":", expand=True)
    if key_parts.shape[1] < 4:
        raise ValueError("Expected key format chrom:pos:ref:alt")

    meta = pd.DataFrame({
        "index": np.arange(len(df), dtype=int),
        "key": df["key"].astype(str),
        "chrom": key_parts[0].map(norm_chrom),
        "pos": pd.to_numeric(key_parts[1], errors="coerce").astype("Int64"),
        "ref": key_parts[2].astype(str).str.upper(),
        "alt": key_parts[3].astype(str).str.upper(),
        "label": labels,
    })

    train_chroms = parse_chrom_list(args.train_chroms)
    val_chroms = parse_chrom_list(args.val_chroms)
    test_chroms = parse_chrom_list(args.test_chroms)
    overlaps = (train_chroms & val_chroms) | (train_chroms & test_chroms) | (val_chroms & test_chroms)
    if overlaps:
        raise ValueError(f"Chromosomes assigned to more than one split: {sorted(overlaps)}")

    meta["split"] = "exclude"
    meta.loc[meta["chrom"].isin(train_chroms), "split"] = "train"
    meta.loc[meta["chrom"].isin(val_chroms), "split"] = "val"
    meta.loc[meta["chrom"].isin(test_chroms), "split"] = "test"

    used = meta[meta["split"].isin(["train", "val", "test"])]
    if used.empty:
        raise ValueError("No rows assigned to train/val/test. Check chromosome naming.")

    for split in ["train", "val", "test"]:
        y = meta.loc[meta["split"] == split, "label"].values
        if len(y) == 0:
            raise ValueError(f"Split {split} has zero rows")
        if len(np.unique(y)) < 2:
            raise ValueError(f"Split {split} lacks both classes: positives={int(y.sum())}, total={len(y)}")

    splits = meta[["index", "split"]].copy()
    splits.to_csv(out_dir / "splits_chrom_reviewer_safe.csv", index=False)
    meta.to_csv(out_dir / "reviewer_metadata.csv", index=False)

    chrom_summary = (
        meta.groupby(["chrom", "split"], dropna=False)
        .agg(n_loci=("label", "size"), n_positive=("label", "sum"))
        .reset_index()
    )
    chrom_summary["n_negative"] = chrom_summary["n_loci"] - chrom_summary["n_positive"]
    chrom_summary["positive_fraction"] = chrom_summary["n_positive"] / chrom_summary["n_loci"]
    chrom_summary.to_csv(out_dir / "chromosome_split_summary.csv", index=False)

    split_summary = (
        meta.groupby("split")
        .agg(n_loci=("label", "size"), n_positive=("label", "sum"))
        .reset_index()
    )
    split_summary["n_negative"] = split_summary["n_loci"] - split_summary["n_positive"]
    split_summary["positive_fraction"] = split_summary["n_positive"] / split_summary["n_loci"]
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)

    print("✅ Reviewer-safe chromosome split created")
    print(f"Output directory: {out_dir.resolve()}")
    print(split_summary.to_string(index=False))
    print("\nUse this split file for reviewer-safe retraining:")
    print(out_dir / "splits_chrom_reviewer_safe.csv")


if __name__ == "__main__":
    main()
