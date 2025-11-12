#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# Set pandas option to handle future warning (optional)
pd.set_option('future.no_silent_downcasting', True)

root = Path("data_root")

# Read UMAP and RTIM data
umap = pd.read_csv(
    root / "umap.tab",
    sep="\t",
    header=None,
    names=["key", "size", "covered", "sum", "MAP"]
)
rtim = pd.read_csv(
    root / "rtim.tab",
    sep="\t",
    header=None,
    names=["key", "size", "covered", "sum", "RTIM"]
)

# Merge UMAP and RTIM data
bw = umap[["key", "MAP"]].merge(rtim[["key", "RTIM"]], on="key", how="outer")

# Read or create QC BAM data
bam_path = root / "qc_bam.csv"
if bam_path.exists():
    qc = pd.read_csv(bam_path)
    # Ensure numeric columns (handle potential strings/objects)
    numeric_cols = ["DP", "AD", "VAF", "MQ", "SB"]
    qc[numeric_cols] = qc[numeric_cols].apply(pd.to_numeric, errors="coerce")
else:
    qc = pd.DataFrame(columns=["key", "DP", "AD", "VAF", "MQ", "SB"])

# Merge all data and fill NA with 0.0 (now with proper type handling)
out = (
    bw.merge(qc, on="key", how="outer")
    .fillna(0.0)
    .infer_objects(copy=False)  # Handle type conversion explicitly
)

# Select and order columns
out = out[["key", "DP", "AD", "VAF", "MQ", "SB", "MAP", "RTIM"]]

# Write output
out.to_csv(root / "qc_merged.csv", index=False)
print("Wrote:", root / "qc_merged.csv")