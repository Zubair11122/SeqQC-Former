#!/usr/bin/env python3

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pysam
from pyfaidx import Fasta
import gc
import traceback


CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])

tumor_bam_path = Path(cfg["tumor_bam"])
normal_bam_path = Path(cfg["normal_bam"]) if cfg.get("normal_bam") else None
reference_fasta = Path(cfg["reference_fasta"])

variants_file = root / "variants_labeled.csv"
out_file = root / "qc_readlevel.csv"

MIN_BASE_QUAL = int(cfg.get("min_base_quality", 13))
MIN_MAPQ = int(cfg.get("min_mapping_quality", 0))

SAVE_EVERY = 5000
REOPEN_EVERY = 10000


def norm_chrom(chrom):
    c = str(chrom).replace("chr", "").strip()
    if c.upper() == "M":
        return "MT"
    return c


def make_key(chrom, pos, ref, alt):
    return f"{norm_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def match_bam_chrom(chrom, bam):
    chrom = norm_chrom(chrom)
    refs = set(bam.references)

    candidates = [chrom, "chr" + chrom]

    if chrom == "MT":
        candidates += ["M", "chrM"]

    for c in candidates:
        if c in refs:
            return c

    return None


class ReferenceHelper:
    def __init__(self, fasta_path):
        self.ref = Fasta(str(fasta_path))
        self.keys = set(self.ref.keys())

    def match_chrom(self, chrom):
        chrom = norm_chrom(chrom)

        candidates = [chrom, "chr" + chrom]

        if chrom == "MT":
            candidates += ["M", "chrM"]

        for c in candidates:
            if c in self.keys:
                return c

        return None

    def homopolymer_length(self, chrom, pos1, flank=10):
        fasta_chrom = self.match_chrom(chrom)

        if fasta_chrom is None:
            return 0

        start0 = max(0, int(pos1) - 1 - flank)
        end0 = int(pos1) + flank

        try:
            seq = str(self.ref[fasta_chrom][start0:end0]).upper()
        except Exception:
            return 0

        if not seq:
            return 0

        max_run = 1
        current = 1

        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1

        return int(max_run)


def default_qc():
    return {
        "DP": 0,
        "AD": 0,
        "VAF": 0.0,
        "MQ": 0.0,
        "BQ": 0.0,
        "SB": 0.0,
        "OrientationBias": 0.0,
        "ClippedFrac": 0.0,
        "MismatchFrac": 0.0,
        "IndelFrac": 0.0,
        "ReadPosBias": 0.0,
    }


def extract_site_qc(bam, chrom, pos1, ref, alt):

    bam_chrom = match_bam_chrom(chrom, bam)

    if bam_chrom is None:
        return default_qc()

    pos0 = int(pos1) - 1
    alt = str(alt).upper()

    dp = 0
    ad = 0

    mq_values = []
    bq_values = []

    alt_fwd = 0
    alt_rev = 0

    clipped_reads = 0
    mismatch_reads = 0
    indel_nearby_reads = 0

    read_pos_alt = []

    try:
        pileups = bam.pileup(
            bam_chrom,
            pos0,
            pos0 + 1,
            truncate=True,
            stepper="samtools",
            max_depth=5000,
            min_base_quality=0,
            ignore_overlaps=True,
            ignore_orphans=True,
        )
    except Exception:
        return default_qc()

    try:
        for pileupcolumn in pileups:

            if pileupcolumn.reference_pos != pos0:
                continue

            for pileupread in pileupcolumn.pileups:

                try:
                    read = pileupread.alignment

                    if (
                        read.is_unmapped
                        or read.is_duplicate
                        or read.is_secondary
                        or read.is_supplementary
                    ):
                        continue

                    if read.mapping_quality < MIN_MAPQ:
                        continue

                    if pileupread.is_del or pileupread.is_refskip:
                        continue

                    qpos = pileupread.query_position

                    if qpos is None:
                        continue

                    if read.query_sequence is None:
                        continue

                    base = read.query_sequence[qpos].upper()

                    if base not in {"A", "C", "G", "T"}:
                        continue

                    if read.query_qualities is not None:
                        bq = int(read.query_qualities[qpos])

                        if bq < MIN_BASE_QUAL:
                            continue
                    else:
                        bq = 0

                    dp += 1

                    mq_values.append(read.mapping_quality)
                    bq_values.append(bq)

                    if read.cigartuples is not None:
                        cigar_ops = [x[0] for x in read.cigartuples]

                        if 4 in cigar_ops or 5 in cigar_ops:
                            clipped_reads += 1

                        if 1 in cigar_ops or 2 in cigar_ops:
                            indel_nearby_reads += 1

                    try:
                        nm = read.get_tag("NM")

                        if nm > 3:
                            mismatch_reads += 1

                    except Exception:
                        pass

                    if base == alt:

                        ad += 1

                        if read.is_reverse:
                            alt_rev += 1
                        else:
                            alt_fwd += 1

                        read_len = read.query_length

                        if read_len and read_len > 0:
                            read_pos_alt.append(qpos / read_len)

                except Exception:
                    continue

    except Exception:
        return default_qc()

    vaf = ad / dp if dp > 0 else 0.0

    mq = float(np.mean(mq_values)) if mq_values else 0.0
    mean_bq = float(np.mean(bq_values)) if bq_values else 0.0

    sb = abs(alt_fwd - alt_rev) / ad if ad > 0 else 0.0

    orientation_bias = max(alt_fwd, alt_rev) / ad if ad > 0 else 0.0

    clipped_fraction = clipped_reads / dp if dp > 0 else 0.0

    mismatch_fraction = mismatch_reads / dp if dp > 0 else 0.0

    indel_fraction = indel_nearby_reads / dp if dp > 0 else 0.0

    if read_pos_alt:
        read_position_bias = abs(float(np.mean(read_pos_alt)) - 0.5)
    else:
        read_position_bias = 0.0

    return {
        "DP": int(dp),
        "AD": int(ad),
        "VAF": float(vaf),
        "MQ": float(mq),
        "BQ": float(mean_bq),
        "SB": float(sb),
        "OrientationBias": float(orientation_bias),
        "ClippedFrac": float(clipped_fraction),
        "MismatchFrac": float(mismatch_fraction),
        "IndelFrac": float(indel_fraction),
        "ReadPosBias": float(read_position_bias),
    }


def paired_features_row(key, tumor_qc, normal_qc, hp):

    delta_vaf = tumor_qc["VAF"] - normal_qc["VAF"]

    germline_support_flag = int(
        normal_qc["AD"] >= 3 and normal_qc["VAF"] >= 0.02
    )

    normal_contamination_flag = int(
        normal_qc["VAF"] > 0 and delta_vaf < 0.05
    )

    return {

        "key": key,

        "DP": tumor_qc["DP"],
        "AD": tumor_qc["AD"],
        "VAF": tumor_qc["VAF"],
        "MQ": tumor_qc["MQ"],
        "SB": tumor_qc["SB"],

        "tumor_depth": tumor_qc["DP"],
        "tumor_alt_count": tumor_qc["AD"],
        "tumor_alt_fraction": tumor_qc["VAF"],
        "tumor_mapq_mean": tumor_qc["MQ"],
        "tumor_baseq_mean": tumor_qc["BQ"],
        "tumor_strand_bias": tumor_qc["SB"],
        "tumor_orientation_bias": tumor_qc["OrientationBias"],
        "tumor_clipped_fraction": tumor_qc["ClippedFrac"],
        "tumor_mismatch_fraction": tumor_qc["MismatchFrac"],
        "tumor_indel_fraction": tumor_qc["IndelFrac"],
        "tumor_read_position_bias": tumor_qc["ReadPosBias"],

        "normal_depth": normal_qc["DP"],
        "normal_alt_count": normal_qc["AD"],
        "normal_alt_fraction": normal_qc["VAF"],
        "normal_mapq_mean": normal_qc["MQ"],
        "normal_baseq_mean": normal_qc["BQ"],
        "normal_strand_bias": normal_qc["SB"],
        "normal_orientation_bias": normal_qc["OrientationBias"],
        "normal_clipped_fraction": normal_qc["ClippedFrac"],
        "normal_mismatch_fraction": normal_qc["MismatchFrac"],
        "normal_indel_fraction": normal_qc["IndelFrac"],
        "normal_read_position_bias": normal_qc["ReadPosBias"],

        "delta_alt_fraction": delta_vaf,
        "germline_support_flag": germline_support_flag,
        "normal_contamination_flag": normal_contamination_flag,
        "homopolymer_length": hp,
    }


def open_bams():

    tumor = pysam.AlignmentFile(str(tumor_bam_path), "rb")

    normal = None

    if normal_bam_path is not None and normal_bam_path.exists():
        normal = pysam.AlignmentFile(str(normal_bam_path), "rb")

    return tumor, normal


def main():

    print("Loading reference FASTA:", reference_fasta)

    ref_helper = ReferenceHelper(reference_fasta)

    print("Reading variants:", variants_file)

    variants = pd.read_csv(variants_file)

    variants["key"] = variants.apply(
        lambda r: make_key(
            r["Chromosome"],
            r["Start_Position"],
            r["Reference_Allele"],
            r["Tumor_Seq_Allele2"],
        ),
        axis=1,
    )

    print("Opening BAMs...")

    tumor_bam, normal_bam = open_bams()

    rows = []

    for i, r in variants.iterrows():

        if i > 0 and i % REOPEN_EVERY == 0:

            print(f"Reopening BAM handles at {i}")

            tumor_bam.close()

            if normal_bam is not None:
                normal_bam.close()

            gc.collect()

            tumor_bam, normal_bam = open_bams()

        try:

            chrom = r["Chromosome"]
            pos = int(r["Start_Position"])
            ref = r["Reference_Allele"]
            alt = r["Tumor_Seq_Allele2"]

            tumor_qc = extract_site_qc(
                tumor_bam,
                chrom,
                pos,
                ref,
                alt,
            )

            normal_qc = default_qc()

            if normal_bam is not None:
                normal_qc = extract_site_qc(
                    normal_bam,
                    chrom,
                    pos,
                    ref,
                    alt,
                )

            hp = ref_helper.homopolymer_length(chrom, pos)

            rows.append(
                paired_features_row(
                    key=r["key"],
                    tumor_qc=tumor_qc,
                    normal_qc=normal_qc,
                    hp=hp,
                )
            )

        except Exception as e:

            print(f"ERROR at row {i}")

            traceback.print_exc()

            continue

        if (i + 1) % SAVE_EVERY == 0:

            print(f"Processed {i+1}/{len(variants)}")

            tmp_df = pd.DataFrame(rows)

            tmp_df.to_csv(out_file, index=False)

            gc.collect()

    tumor_bam.close()

    if normal_bam is not None:
        normal_bam.close()

    df = pd.DataFrame(rows)

    df.to_csv(out_file, index=False)

    print("\nSaved:", out_file)
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()