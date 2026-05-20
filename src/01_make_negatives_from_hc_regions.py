#!/usr/bin/env python3

from pathlib import Path
import random
import gzip
import yaml
import pandas as pd
from pyfaidx import Fasta


CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])
reference_fasta = Path(cfg["reference_fasta"])

# Your files
bed_file = root / "seqc2_truth" / "hc_regions.from_truth.bed"
positive_vcf = root / "seqc2_truth" / "high-confidence_sSNV_in_HC_regions_v1.2.1.vcf.gz"

out_dir = root / "negatives"
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "confirmed_negatives.tsv"

N_NEGATIVES = 50000
random.seed(42)


def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8", errors="replace")


def norm_chrom(chrom):
    return str(chrom).replace("chr", "").strip()


def match_fasta_chrom(chrom, fasta):
    chrom = norm_chrom(chrom)

    if chrom in fasta.keys():
        return chrom

    chr_chrom = "chr" + chrom
    if chr_chrom in fasta.keys():
        return chr_chrom

    return None


def random_alt(ref):
    bases = ["A", "C", "G", "T"]
    return random.choice([b for b in bases if b != ref.upper()])


print("Checking input files...")

if not bed_file.exists():
    raise FileNotFoundError(f"Missing BED file: {bed_file}")

if not positive_vcf.exists():
    raise FileNotFoundError(f"Missing positive VCF file: {positive_vcf}")

if not reference_fasta.exists():
    raise FileNotFoundError(f"Missing reference FASTA: {reference_fasta}")

print("Reading positive SEQC2 truth SNVs...")

positive_positions = set()

with open_text(positive_vcf) as fh:
    for line in fh:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 5:
            continue

        chrom = norm_chrom(fields[0])
        pos = int(fields[1])
        ref = fields[3]
        alts = fields[4].split(",")

        for alt in alts:
            if len(ref) == 1 and len(alt) == 1:
                positive_positions.add((chrom, pos))

print("Positive SNV positions:", len(positive_positions))


print("Reading high-confidence BED regions...")

regions = []

with open_text(bed_file) as fh:
    for line in fh:
        if not line.strip() or line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        chrom = norm_chrom(fields[0])
        start0 = int(fields[1])
        end0 = int(fields[2])

        if end0 > start0:
            regions.append((chrom, start0, end0))

print("HC regions:", len(regions))


print("Opening reference FASTA with pyfaidx...")

fasta = Fasta(str(reference_fasta), rebuild=False)

rows = []
seen = set()
attempts = 0
max_attempts = N_NEGATIVES * 200

print("Sampling negative loci...")

while len(rows) < N_NEGATIVES and attempts < max_attempts:
    attempts += 1

    chrom, start0, end0 = random.choice(regions)

    # BED = 0-based, VCF position = 1-based
    pos1 = random.randint(start0 + 1, end0)

    if (chrom, pos1) in positive_positions:
        continue

    if (chrom, pos1) in seen:
        continue

    fasta_chrom = match_fasta_chrom(chrom, fasta)

    if fasta_chrom is None:
        continue

    try:
        # pyfaidx uses 0-based Python-style slicing
        ref = str(fasta[fasta_chrom][pos1 - 1:pos1]).upper()
    except Exception:
        continue

    if ref not in {"A", "C", "G", "T"}:
        continue

    alt = random_alt(ref)

    rows.append([chrom, pos1, ref, alt])
    seen.add((chrom, pos1))

    if len(rows) % 1000 == 0:
        print("Sampled:", len(rows))

if len(rows) == 0:
    raise SystemExit("No negatives sampled. Check chromosome names in BED and FASTA.")

df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])

df.to_csv(out_file, sep="\t", index=False)

print("\nSaved:", out_file)
print("Negative count:", len(df))
print(df.head())