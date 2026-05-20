#!/usr/bin/env python3
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pyBigWig

# Load config
CFG = Path("config.yaml")
with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])
variants_file = root / "variants_labeled.csv"
umap_bw_path = Path(cfg["umap_bw"])
rtim_bw_path = Path(cfg["rtim_bw"])
out_file = root / "qc_map_rtim.csv"

WINDOWS_BED = [21, 50, 100, 200]
FEATURE_WINDOW = int(cfg.get("map_rtim_window", 21))

def norm_chrom(chrom):
    return str(chrom).replace("chr","").strip()

def make_key(chrom,pos,ref,alt):
    return f"{norm_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"

def match_bw_chrom(chrom,bw):
    chrom = norm_chrom(chrom)
    chroms = bw.chroms()
    if chrom in chroms: return chrom
    if "chr"+chrom in chroms: return "chr"+chrom
    return None

def bw_mean_window(bw, chrom, pos1, window=FEATURE_WINDOW):
    bw_chrom = match_bw_chrom(chrom,bw)
    if bw_chrom is None:
        return 0.0
    pos0 = int(pos1) - 1
    start = max(0,pos0-window//2)
    end = pos0+window//2+1
    try:
        values = bw.values(bw_chrom, start, end)
        values = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.mean(values)) if values else 0.0
    except Exception:
        return 0.0

# Read variants
variants = pd.read_csv(variants_file)
variants["key"] = variants.apply(lambda r: make_key(
    r["Chromosome"], r["Start_Position"], r["Reference_Allele"], r["Tumor_Seq_Allele2"]), axis=1)

# Multi-window BED output
for W in WINDOWS_BED:
    bed = variants.copy()
    bed["start"] = (bed["Start_Position"] - W).clip(lower=0)
    bed["end"] = bed["Start_Position"] + W
    bed_file = root / f"sites_win_{W}bp.bed"
    bed[["Chromosome","start","end","key"]].to_csv(bed_file, sep="\t", header=False, index=False)
    print(f"Saved BED window ({W}bp): {bed_file}")

# Open BigWigs
umap_bw = pyBigWig.open(str(umap_bw_path))
rtim_bw = pyBigWig.open(str(rtim_bw_path))

# Compute MAP/RTIM features
rows=[]
for i,r in variants.iterrows():
    chrom = r["Chromosome"]
    pos = r["Start_Position"]
    map_val = bw_mean_window(umap_bw, chrom, pos, FEATURE_WINDOW)
    rtim_val = bw_mean_window(rtim_bw, chrom, pos, FEATURE_WINDOW)
    rows.append({
        "key": r["key"],
        "Chromosome": norm_chrom(chrom),
        "Start_Position": int(pos),
        "seqc2_positive": int(r.get("seqc2_positive",0)),
        "MAP": map_val,
        "RTIM": rtim_val,
        "MAP_missing": int(pd.isna(map_val)),
        "RTIM_missing": int(pd.isna(rtim_val))
    })
    if (i+1) % 5000 == 0:
        print(f"Processed {i+1}/{len(variants)}")

umap_bw.close()
rtim_bw.close()

df = pd.DataFrame(rows)
df["MAP"] = df["MAP"].fillna(0.0)
df["RTIM"] = df["RTIM"].fillna(df["RTIM"].median(skipna=True))
df.to_csv(out_file,index=False)
print("Saved QC/Map features:", out_file)