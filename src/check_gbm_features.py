#!/usr/bin/env python3
"""
Check GBM features for validity
"""
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

root = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/data_root")
features_h5 = root / "gbm" / "gbm_features.h5"
sites_file = root / "gbm" / "gbm_sites.csv"

print("="*60)
print("GBM Features Diagnostic Check")
print("="*60)

# Check HDF5 file
print(f"\n📁 HDF5 File: {features_h5}")
if not features_h5.exists():
    print("   ❌ File not found!")
    exit(1)

file_size_mb = features_h5.stat().st_size / 1e6
print(f"   File size: {file_size_mb:.2f} MB")

with h5py.File(features_h5, "r") as f:
    print(f"\n📊 Datasets in HDF5:")
    for key in f.keys():
        shape = f[key].shape
        dtype = f[key].dtype
        print(f"   {key}: {shape} {dtype}")
    
    # Check sequences
    sequences = f["sequences"][:]
    print(f"\n🧬 Sequences check:")
    print(f"   Shape: {sequences.shape}")
    print(f"   Data type: {sequences.dtype}")
    print(f"   Min value: {sequences.min():.4f}")
    print(f"   Max value: {sequences.max():.4f}")
    print(f"   Mean value: {sequences.mean():.4f}")
    
    # Check if sequences are all zeros
    zero_sequences = np.all(sequences == 0, axis=(1, 2))
    n_zero = zero_sequences.sum()
    zero_percent = n_zero/len(sequences)*100
    print(f"   All-zero sequences: {n_zero} / {len(sequences)} ({zero_percent:.1f}%)")
    
    # Check sample sequence
    print(f"\n🔍 Sample sequence stats (first variant):")
    sample_seq = sequences[0]
    print(f"   Shape: {sample_seq.shape}")
    channel_sums = [sample_seq[0].sum(), sample_seq[1].sum(), sample_seq[2].sum(), sample_seq[3].sum()]
    print(f"   Channel sums: A={channel_sums[0]:.1f}, C={channel_sums[1]:.1f}, G={channel_sums[2]:.1f}, T={channel_sums[3]:.1f}")
    total_bases = sum(channel_sums)
    print(f"   Total bases represented: {total_bases:.1f} (expected ~501)")
    
    # Check QC features
    qc_features = f["qc_features"][:]
    print(f"\n📊 QC features check:")
    print(f"   Shape: {qc_features.shape}")
    print(f"   Mean per column: {qc_features.mean(axis=0)}")

# Check sites file
print(f"\n📁 Sites file: {sites_file}")
sites = pd.read_csv(sites_file)
print(f"   Total variants: {len(sites):,}")
print(f"   Chromosomes present: {sites['Chromosome'].nunique()}")
print(f"   First 10 chromosomes: {sites['Chromosome'].head(10).tolist()}")

# Check if chromosomes have 'chr' prefix
has_chr_prefix = sites['Chromosome'].astype(str).str.startswith('chr').any()
print(f"   Has 'chr' prefix: {has_chr_prefix}")

print("\n" + "="*60)
if file_size_mb < 100:
    print("⚠️ WARNING: File is too small (<100 MB)!")
    print("   This suggests sequences are not being extracted correctly.")
    print("   Chromosome name mismatch is the likely cause.")
else:
    print("✅ File size looks good.")
print("="*60)