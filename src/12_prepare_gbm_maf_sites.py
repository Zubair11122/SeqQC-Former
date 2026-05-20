#!/usr/bin/env python3

from pathlib import Path
import gzip
import yaml
import pandas as pd


CFG = Path("config.yaml")

with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["data_root"])

# Add this to config.yaml if you want:
# gbm_maf_dir: /path/to/GBM_MAF_folder
gbm_maf_dir = Path(cfg.get("gbm_maf_dir", root / "gbm_maf"))

out_dir = root / "gbm"
out_dir.mkdir(parents=True, exist_ok=True)

out_sites = out_dir / "gbm_sites.csv"
out_variants_like = out_dir / "gbm_variants_labeled_like_seqc2.csv"
out_manifest = out_dir / "gbm_maf_manifest.csv"


def norm_chrom(chrom):
    c = str(chrom).replace("chr", "").strip()
    if c.upper() == "M":
        return "MT"
    return c


def make_key(chrom, pos, ref, alt):
    return f"{norm_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def read_maf(path):
    # MAF can have comment/header metadata lines beginning with #.
    with open_text(path) as fh:
        lines = [line for line in fh if not line.startswith("#") and line.strip()]

    if not lines:
        return pd.DataFrame()

    # Write-free parse.
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)


def choose_alt(row):
    # Prefer Tumor_Seq_Allele2, but handle MAFs where allele1 carries the non-reference allele.
    ref = str(row.get("Reference_Allele", "")).upper()
    a1 = str(row.get("Tumor_Seq_Allele1", "")).upper()
    a2 = str(row.get("Tumor_Seq_Allele2", "")).upper()

    if a2 and a2 != ref and a2 not in {"NAN", "NONE", "."}:
        return a2
    if a1 and a1 != ref and a1 not in {"NAN", "NONE", "."}:
        return a1
    return a2


def main():
    if not gbm_maf_dir.exists():
        raise FileNotFoundError(
            f"GBM MAF folder not found: {gbm_maf_dir}\n"
            "Either create data_root/gbm_maf or add gbm_maf_dir: /path/to/folder in config.yaml"
        )

    maf_files = []
    for pattern in ["*.maf", "*.maf.gz", "*.tsv", "*.tsv.gz", "*.txt", "*.txt.gz"]:
        maf_files.extend(sorted(gbm_maf_dir.glob(pattern)))

    if not maf_files:
        raise FileNotFoundError(f"No MAF/TSV/TXT files found in {gbm_maf_dir}")

    all_rows = []
    manifest = []

    required = ["Chromosome", "Start_Position", "Reference_Allele"]

    for maf in maf_files:
        print("Reading:", maf)
        df = read_maf(maf)

        if df.empty:
            print("  WARNING: empty or unreadable file, skipped")
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  WARNING: missing required columns {missing}, skipped")
            continue

        df = df.copy()
        df["source_maf"] = maf.name
        df["Tumor_Seq_Allele2"] = df.apply(choose_alt, axis=1)

        # Keep only simple SNVs for the same model type as your current SEQC2 SNV pipeline.
        df["Reference_Allele"] = df["Reference_Allele"].astype(str).str.upper()
        df["Tumor_Seq_Allele2"] = df["Tumor_Seq_Allele2"].astype(str).str.upper()

        snv = (
            df["Reference_Allele"].isin(["A", "C", "G", "T"])
            & df["Tumor_Seq_Allele2"].isin(["A", "C", "G", "T"])
            & (df["Reference_Allele"] != df["Tumor_Seq_Allele2"])
        )

        before = len(df)
        df = df.loc[snv].copy()
        after = len(df)

        if "End_Position" not in df.columns:
            df["End_Position"] = df["Start_Position"]

        if "Tumor_Sample_Barcode" not in df.columns:
            df["Tumor_Sample_Barcode"] = maf.stem

        if "Matched_Norm_Sample_Barcode" not in df.columns:
            df["Matched_Norm_Sample_Barcode"] = ""

        df["Chromosome"] = df["Chromosome"].map(norm_chrom)
        df["Start_Position"] = df["Start_Position"].astype(int)
        df["End_Position"] = df["End_Position"].astype(int)

        df["key"] = df.apply(
            lambda r: make_key(
                r["Chromosome"],
                r["Start_Position"],
                r["Reference_Allele"],
                r["Tumor_Seq_Allele2"],
            ),
            axis=1,
        )

        manifest.append(
            {
                "source_maf": maf.name,
                "rows_total": before,
                "snv_rows_kept": after,
            }
        )

        keep_cols = [
            "key",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "Tumor_Sample_Barcode",
            "Matched_Norm_Sample_Barcode",
            "source_maf",
        ]

        # Keep additional useful MAF columns if present.
        optional = [
            "Hugo_Symbol",
            "Variant_Classification",
            "Variant_Type",
            "t_depth",
            "t_ref_count",
            "t_alt_count",
            "n_depth",
            "n_ref_count",
            "n_alt_count",
            "FILTER",
            "CENTERS",
            "NCBI_Build",
        ]

        keep_cols += [c for c in optional if c in df.columns and c not in keep_cols]
        all_rows.append(df[keep_cols])

    if not all_rows:
        raise RuntimeError("No usable SNV rows were found in the GBM MAF folder.")

    sites = pd.concat(all_rows, ignore_index=True)
    sites = sites.drop_duplicates(subset=["key", "Tumor_Sample_Barcode", "source_maf"]).reset_index(drop=True)

    # Variant table compatible with feature extraction scripts.
    variants_like = sites.copy()
    variants_like["label"] = -1  # Unknown label; GBM is external application, not training truth.

    sites.to_csv(out_sites, index=False)
    variants_like.to_csv(out_variants_like, index=False)
    pd.DataFrame(manifest).to_csv(out_manifest, index=False)

    print("\nSaved:", out_sites)
    print("Saved:", out_variants_like)
    print("Saved:", out_manifest)
    print("GBM SNV sites:", len(sites))
    print(sites.head())


if __name__ == "__main__":
    main()
