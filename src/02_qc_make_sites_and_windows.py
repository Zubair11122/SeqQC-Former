#!/usr/bin/env python3
from pathlib import Path
import yaml, pandas as pd

# Load config (adjust path if needed)
CFG = Path("/home/zubair/Project/config.yaml")  # Full explicit path
cfg = yaml.safe_load(open(CFG))
root = Path(cfg["data_root"])

# Read and preprocess variants
try:
    df = pd.read_pickle(root / "variants_labeled.pkl")
except FileNotFoundError:
    raise SystemExit(f"Error: {root}/variants_labeled.pkl not found!")

df = df.rename(columns={
    "Chromosome": "chrom",
    "Start_Position": "pos",
    "Reference_Allele": "ref",
    "Tumor_Seq_Allele2": "alt"
})
df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
df["key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]

# Save sites.csv
df[["chrom", "pos", "ref", "alt"]].to_csv(root / "sites.csv", index=False)

# Generate BED windows
W = int(cfg.get("window_bp", 100))
if W <= 0:
    raise ValueError("window_bp must be > 0")

bed = df.copy()
bed["start"] = (bed["pos"] - 1 - W).clip(lower=0)  # 0-based start
bed["end"] = bed["pos"] + W  # 1-based end
bed[["chrom", "start", "end", "key"]].to_csv(
    root / "sites_win.bed",
    sep="\t",
    header=False,
    index=False
)

print("Wrote:", root / "sites.csv", "and", root / "sites_win.bed")