#!/usr/bin/env python3
"""
11A_extract_gbm_real_qc_from_bams.py

Extract real read-level QC features for GBM variants from tumor/normal BAM or CRAM files.

Input:
  1. data_root/gbm/gbm_sites.csv
     Created by 10_prepare_gbm_maf_sites.py

  2. A BAM/CRAM manifest CSV, default:
     data_root/gbm/gbm_bam_manifest.csv

     Required columns:
       Tumor_Sample_Barcode,tumor_bam

     Optional columns:
       normal_bam

     Example:
       Tumor_Sample_Barcode,tumor_bam,normal_bam
       TCGA-06-5416-01A-01D-1486-08,/path/tumor1.bam,/path/normal1.bam
       TCGA-19-5956-01A-11D-1696-08,/path/tumor2.cram,/path/normal2.cram

Output:
  data_root/gbm/gbm_qc_readlevel.csv

Notes:
  - Works with BAM and CRAM.
  - BAM needs .bai index.
  - CRAM needs .crai index and reference_fasta in config.yaml.
  - The script groups variants by Tumor_Sample_Barcode and opens the matching BAM/CRAM.
  - It creates the exact 11 QC features expected by your current trained model:
      DP, AD, VAF, MQ, SB,
      tumor_strand_bias, tumor_orientation_bias,
      tumor_clipped_fraction, tumor_mismatch_fraction,
      normal_alt_fraction, germline_support_flag
  - It also writes extra useful read-level columns for future model versions.
"""

from pathlib import Path
import sys
import gc
import traceback
import yaml
import numpy as np
import pandas as pd
import pysam
from pyfaidx import Fasta
from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------
CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])
gbm_dir = root / "gbm"

sites_file = gbm_dir / "gbm_sites.csv"

# Default manifest location, can override in config.yaml:
# gbm_bam_manifest: /path/to/gbm_bam_manifest.csv
manifest_file = Path(cfg.get("gbm_bam_manifest", gbm_dir / "gbm_bam_manifest.csv"))

reference_fasta = Path(cfg["reference_fasta"])

out_file = gbm_dir / "gbm_qc_readlevel.csv"
progress_file = gbm_dir / "gbm_qc_readlevel.progress.csv"
missing_file = gbm_dir / "gbm_missing_bam_samples.csv"
log_file = gbm_dir / "gbm_qc_extraction_log.txt"

MIN_BASE_QUAL = int(cfg.get("min_base_quality", 13))
MIN_MAPQ = int(cfg.get("min_mapping_quality", 0))
MAX_DEPTH = int(cfg.get("pileup_max_depth", 5000))
SAVE_EVERY_SAMPLES = int(cfg.get("gbm_qc_save_every_samples", 1))

# For CRAM, pysam needs reference filename.
# For BAM, reference_filename is usually not required but harmless.
REFERENCE_FOR_ALIGNMENT = str(reference_fasta) if reference_fasta.exists() else None


# ----------------------------
# Utility functions
# ----------------------------
def norm_chrom(chrom):
    c = str(chrom).replace("chr", "").strip()
    if c.upper() in {"M", "MT"}:
        return "MT"
    return c


def make_key(chrom, pos, ref, alt):
    return f"{norm_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def is_missing_path(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "."}


def check_alignment_index(path: Path):
    """
    Check common BAM/CRAM index filenames.
    Does not stop execution if index is missing, because some paths may use remote/index configs,
    but it prints a warning.
    """
    p = Path(path)
    candidates = []

    if p.suffix.lower() == ".bam":
        candidates = [
            Path(str(p) + ".bai"),
            p.with_suffix(".bai"),
        ]
    elif p.suffix.lower() == ".cram":
        candidates = [
            Path(str(p) + ".crai"),
            p.with_suffix(".crai"),
        ]

    if candidates and not any(c.exists() for c in candidates):
        print(f"  ⚠️ Warning: index not found for {p}")
        print(f"     Expected one of: {', '.join(str(c) for c in candidates)}")


def open_alignment(path):
    """
    Open BAM or CRAM.
    CRAM requires reference_fasta.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Alignment file not found: {path}")

    check_alignment_index(path)

    suffix = path.suffix.lower()

    if suffix == ".cram":
        if REFERENCE_FOR_ALIGNMENT is None:
            raise FileNotFoundError(
                "CRAM input requires reference_fasta in config.yaml and the file must exist."
            )
        return pysam.AlignmentFile(str(path), "rc", reference_filename=REFERENCE_FOR_ALIGNMENT)

    # BAM or unknown extension
    return pysam.AlignmentFile(str(path), "rb", reference_filename=REFERENCE_FOR_ALIGNMENT)


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
        self.ref = Fasta(str(fasta_path), rebuild=False)
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
    """
    Extract read-level QC at one SNV site from one BAM/CRAM.
    """
    if bam is None:
        return default_qc()

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
            max_depth=MAX_DEPTH,
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

                        # 4 = soft clip, 5 = hard clip
                        if 4 in cigar_ops or 5 in cigar_ops:
                            clipped_reads += 1

                        # 1 = insertion, 2 = deletion
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

    # Strand bias: 0 means balanced/no alt support, 1 means all alt reads on one strand
    sb = abs(alt_fwd - alt_rev) / ad if ad > 0 else 0.0

    # Orientation bias: 0.5 means balanced, 1 means all alt reads one orientation
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


def paired_features_row(site_row, tumor_qc, normal_qc, hp):
    """
    Combine tumor and normal QC into one output row.
    """
    key = str(site_row["key"])

    delta_vaf = tumor_qc["VAF"] - normal_qc["VAF"]

    germline_support_flag = int(
        normal_qc["AD"] >= 3 and normal_qc["VAF"] >= 0.02
    )

    normal_contamination_flag = int(
        normal_qc["VAF"] > 0 and delta_vaf < 0.05
    )

    return {
        # Required identity columns
        "key": key,
        "Chromosome": norm_chrom(site_row["Chromosome"]),
        "Start_Position": int(site_row["Start_Position"]),
        "End_Position": int(site_row.get("End_Position", site_row["Start_Position"])),
        "Reference_Allele": str(site_row["Reference_Allele"]).upper(),
        "Tumor_Seq_Allele2": str(site_row["Tumor_Seq_Allele2"]).upper(),
        "Tumor_Sample_Barcode": site_row.get("Tumor_Sample_Barcode", ""),

        # Exact 11 QC columns expected by your current trained model
        "DP": tumor_qc["DP"],
        "AD": tumor_qc["AD"],
        "VAF": tumor_qc["VAF"],
        "MQ": tumor_qc["MQ"],
        "SB": tumor_qc["SB"],
        "tumor_strand_bias": tumor_qc["SB"],
        "tumor_orientation_bias": tumor_qc["OrientationBias"],
        "tumor_clipped_fraction": tumor_qc["ClippedFrac"],
        "tumor_mismatch_fraction": tumor_qc["MismatchFrac"],
        "normal_alt_fraction": normal_qc["VAF"],
        "germline_support_flag": germline_support_flag,

        # Extra useful real-QC columns
        "tumor_depth": tumor_qc["DP"],
        "tumor_alt_count": tumor_qc["AD"],
        "tumor_alt_fraction": tumor_qc["VAF"],
        "tumor_mapq_mean": tumor_qc["MQ"],
        "tumor_baseq_mean": tumor_qc["BQ"],
        "tumor_indel_fraction": tumor_qc["IndelFrac"],
        "tumor_read_position_bias": tumor_qc["ReadPosBias"],

        "normal_depth": normal_qc["DP"],
        "normal_alt_count": normal_qc["AD"],
        "normal_mapq_mean": normal_qc["MQ"],
        "normal_baseq_mean": normal_qc["BQ"],
        "normal_strand_bias": normal_qc["SB"],
        "normal_orientation_bias": normal_qc["OrientationBias"],
        "normal_clipped_fraction": normal_qc["ClippedFrac"],
        "normal_mismatch_fraction": normal_qc["MismatchFrac"],
        "normal_indel_fraction": normal_qc["IndelFrac"],
        "normal_read_position_bias": normal_qc["ReadPosBias"],

        "delta_alt_fraction": delta_vaf,
        "normal_contamination_flag": normal_contamination_flag,
        "homopolymer_length": hp,
    }


def load_manifest(path: Path):
    """
    Load sample-to-BAM manifest.
    """
    require_file(path, "GBM BAM manifest")

    manifest = pd.read_csv(path)

    required = ["Tumor_Sample_Barcode", "tumor_bam"]
    missing = [c for c in required if c not in manifest.columns]

    if missing:
        raise KeyError(
            f"Manifest is missing required columns: {missing}\n"
            "Required columns: Tumor_Sample_Barcode,tumor_bam\n"
            "Optional column: normal_bam"
        )

    if "normal_bam" not in manifest.columns:
        manifest["normal_bam"] = ""

    manifest["Tumor_Sample_Barcode"] = manifest["Tumor_Sample_Barcode"].astype(str)

    # Remove duplicate sample rows, keep first
    dup = manifest["Tumor_Sample_Barcode"].duplicated().sum()
    if dup > 0:
        print(f"⚠️ Warning: manifest has {dup} duplicated Tumor_Sample_Barcode values; keeping first.")
        manifest = manifest.drop_duplicates("Tumor_Sample_Barcode", keep="first")

    return manifest.set_index("Tumor_Sample_Barcode").to_dict(orient="index")


def prepare_sites(path: Path):
    """
    Load GBM sites and ensure full key exists.
    """
    require_file(path, "GBM sites CSV")

    sites = pd.read_csv(path)

    required = [
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
    ]

    missing = [c for c in required if c not in sites.columns]
    if missing:
        raise KeyError(f"gbm_sites.csv missing required columns: {missing}")

    if "End_Position" not in sites.columns:
        sites["End_Position"] = sites["Start_Position"]

    if "key" not in sites.columns:
        sites["key"] = sites.apply(
            lambda r: make_key(
                r["Chromosome"],
                r["Start_Position"],
                r["Reference_Allele"],
                r["Tumor_Seq_Allele2"],
            ),
            axis=1,
        )
    else:
        # Normalize just to string
        sites["key"] = sites["key"].astype(str)

    sites["Tumor_Sample_Barcode"] = sites["Tumor_Sample_Barcode"].astype(str)

    return sites


def append_log(message):
    with open(log_file, "a") as f:
        f.write(message.rstrip() + "\n")


# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 80)
    print("GBM real read-level QC extraction from BAM/CRAM")
    print("=" * 80)

    require_file(reference_fasta, "Reference FASTA")
    require_file(sites_file, "GBM sites CSV")

    print(f"Sites file:       {sites_file}")
    print(f"Manifest file:    {manifest_file}")
    print(f"Reference FASTA:  {reference_fasta}")
    print(f"Output QC file:   {out_file}")
    print(f"Min base quality: {MIN_BASE_QUAL}")
    print(f"Min mapping qual: {MIN_MAPQ}")
    print(f"Pileup max depth: {MAX_DEPTH}")

    # Reset log
    log_file.write_text("")
    append_log("GBM real read-level QC extraction log")
    append_log("=" * 80)

    print("\nLoading sites...")
    sites = prepare_sites(sites_file)
    print(f"  Total GBM site rows: {len(sites):,}")
    print(f"  Samples in sites: {sites['Tumor_Sample_Barcode'].nunique():,}")

    print("\nLoading BAM/CRAM manifest...")
    manifest = load_manifest(manifest_file)
    print(f"  Samples in manifest: {len(manifest):,}")

    missing_samples = sorted(set(sites["Tumor_Sample_Barcode"]) - set(manifest.keys()))
    if missing_samples:
        pd.DataFrame({"Tumor_Sample_Barcode": missing_samples}).to_csv(missing_file, index=False)
        print(f"\n⚠️ Missing BAM/CRAM for {len(missing_samples):,} samples.")
        print(f"  Missing sample list saved to: {missing_file}")
        print("  These samples will be skipped.")
        append_log(f"Missing samples: {len(missing_samples)}")

    print("\nOpening reference helper...")
    ref_helper = ReferenceHelper(reference_fasta)

    rows = []
    processed_samples = 0
    skipped_samples = 0
    failed_samples = 0

    grouped = list(sites.groupby("Tumor_Sample_Barcode", sort=False))

    print("\nExtracting read-level QC by sample...")

    for sample_id, sample_sites in tqdm(grouped, desc="Samples"):
        sample_id = str(sample_id)

        if sample_id not in manifest:
            skipped_samples += 1
            continue

        record = manifest[sample_id]
        tumor_bam_path = record.get("tumor_bam", "")

        if is_missing_path(tumor_bam_path):
            skipped_samples += 1
            append_log(f"SKIP {sample_id}: missing tumor_bam path")
            continue

        normal_bam_path = record.get("normal_bam", "")
        has_normal = not is_missing_path(normal_bam_path)

        tumor_bam = None
        normal_bam = None

        try:
            print(f"\nSample: {sample_id}")
            print(f"  Variants: {len(sample_sites):,}")
            print(f"  Tumor: {tumor_bam_path}")
            if has_normal:
                print(f"  Normal: {normal_bam_path}")
            else:
                print("  Normal: not available")

            tumor_bam = open_alignment(tumor_bam_path)

            if has_normal:
                normal_bam = open_alignment(normal_bam_path)

            sample_rows = []

            for _, r in tqdm(
                sample_sites.iterrows(),
                total=len(sample_sites),
                desc=f"{sample_id}",
                leave=False,
            ):
                chrom = r["Chromosome"]
                pos = int(r["Start_Position"])
                ref = str(r["Reference_Allele"]).upper()
                alt = str(r["Tumor_Seq_Allele2"]).upper()

                tumor_qc = extract_site_qc(
                    tumor_bam,
                    chrom,
                    pos,
                    ref,
                    alt,
                )

                if normal_bam is not None:
                    normal_qc = extract_site_qc(
                        normal_bam,
                        chrom,
                        pos,
                        ref,
                        alt,
                    )
                else:
                    normal_qc = default_qc()

                hp = ref_helper.homopolymer_length(chrom, pos)

                sample_rows.append(
                    paired_features_row(
                        site_row=r,
                        tumor_qc=tumor_qc,
                        normal_qc=normal_qc,
                        hp=hp,
                    )
                )

            rows.extend(sample_rows)
            processed_samples += 1

            append_log(
                f"OK {sample_id}: variants={len(sample_sites)}, "
                f"tumor={tumor_bam_path}, normal={normal_bam_path if has_normal else 'NA'}"
            )

        except Exception as e:
            failed_samples += 1
            append_log(f"FAIL {sample_id}: {e}")
            append_log(traceback.format_exc())
            print(f"\n❌ Failed sample: {sample_id}")
            traceback.print_exc()

        finally:
            try:
                if tumor_bam is not None:
                    tumor_bam.close()
            except Exception:
                pass

            try:
                if normal_bam is not None:
                    normal_bam.close()
            except Exception:
                pass

            gc.collect()

        if processed_samples > 0 and processed_samples % SAVE_EVERY_SAMPLES == 0:
            tmp = pd.DataFrame(rows)
            tmp.to_csv(progress_file, index=False)
            print(f"  Progress saved: {progress_file} ({len(tmp):,} rows)")

    if not rows:
        raise RuntimeError(
            "No QC rows were produced. Check manifest sample IDs and BAM/CRAM paths."
        )

    qc = pd.DataFrame(rows)

    # Preserve the original gbm_sites.csv row order as much as possible.
    # Since we grouped by sample, reorder by a stable row_id from original sites.
    # Add original row_id temporarily.
    order_df = sites[["key", "Tumor_Sample_Barcode"]].copy()
    order_df["_original_order"] = np.arange(len(order_df))

    qc = qc.merge(
        order_df,
        on=["key", "Tumor_Sample_Barcode"],
        how="left",
        validate="many_to_many",
    )

    qc = qc.sort_values("_original_order", na_position="last").drop(columns=["_original_order"])

    qc.to_csv(out_file, index=False)

    print("\n" + "=" * 80)
    print("✅ Real read-level GBM QC extraction completed")
    print("=" * 80)
    print(f"Output: {out_file}")
    print(f"Rows written: {len(qc):,}")
    print(f"Samples processed: {processed_samples:,}")
    print(f"Samples skipped: {skipped_samples:,}")
    print(f"Samples failed: {failed_samples:,}")

    print("\nQC summary:")
    model_cols = [
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

    print(qc[model_cols].describe().T)

    print("\nMissing/zero depth checks:")
    print(f"  Tumor DP == 0: {(qc['DP'] == 0).sum():,} / {len(qc):,} ({(qc['DP'] == 0).mean():.2%})")
    print(f"  Tumor AD == 0: {(qc['AD'] == 0).sum():,} / {len(qc):,} ({(qc['AD'] == 0).mean():.2%})")
    print(f"  Normal available rows with normal_depth > 0: {(qc['normal_depth'] > 0).sum():,}")

    print(f"\nLog file: {log_file}")
    if missing_samples:
        print(f"Missing sample list: {missing_file}")

    print("\nNext step:")
    print("  Update/modify 11_extract_gbm_features.py to read gbm_qc_readlevel.csv")
    print("  instead of MAF-derived approximate QC features.")


if __name__ == "__main__":
    main()
