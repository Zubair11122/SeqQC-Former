#!/usr/bin/env python3

import gzip
import pandas as pd
from pathlib import Path
import yaml


CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])

truth_vcf = root / "seqc2_truth" / "high-confidence_sSNV_in_HC_regions_v1.2.1.vcf.gz"
neg_file = root / "negatives" / "confirmed_negatives.tsv"

out_csv = root / "variants_labeled.csv"
out_pkl = root / "variants_labeled.pkl"


def open_text(path):
    path = str(path)

    if path.endswith(".gz"):
        return gzip.open(path, "rt")

    return open(path, "r", encoding="utf-8", errors="replace")


def norm_chrom(c):
    return str(c).replace("chr", "").strip()


print("Reading positives from:", truth_vcf)

pos = []

with open_text(truth_vcf) as fh:
    for line in fh:
        if line.startswith("#"):
            continue

        f = line.strip().split("\t")

        chrom = norm_chrom(f[0])
        posi = int(f[1])
        ref = f[3]

        for alt in f[4].split(","):
            if len(ref) == 1 and len(alt) == 1:
                pos.append(
                    [
                        chrom,
                        posi,
                        ref,
                        alt,
                        1,
                        "SEQC2_truth"
                    ]
                )

dfpos = pd.DataFrame(
    pos,
    columns=[
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "seqc2_positive",
        "source"
    ]
)

print("Positive SNVs:", len(dfpos))

print("Reading negatives from:", neg_file)

neg = pd.read_csv(neg_file, sep="\t")

neg = neg.rename(
    columns={
        "chrom": "Chromosome",
        "pos": "Start_Position",
        "ref": "Reference_Allele",
        "alt": "Tumor_Seq_Allele2"
    }
)

neg["seqc2_positive"] = 0
neg["source"] = "sampled_HC_nontruth_negative"

df = pd.concat([dfpos, neg], ignore_index=True)

df["Chromosome"] = df["Chromosome"].map(norm_chrom)

# If exact same variant appears twice, keep positive first
df = df.sort_values("seqc2_positive", ascending=False)

df = df.drop_duplicates(
    subset=[
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2"
    ],
    keep="first"
)

df = df.reset_index(drop=True)

df.to_csv(out_csv, index=False)
df.to_pickle(out_pkl)

print("\nSaved:")
print(out_csv)
print(out_pkl)

print("\nLabel counts:")
print(df["seqc2_positive"].value_counts())

print("\nSource counts:")
print(df["source"].value_counts())

print("\nPreview:")
print(df.head())