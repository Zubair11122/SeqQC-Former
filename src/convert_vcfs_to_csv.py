from pathlib import Path
import gzip
import pandas as pd

PROJECT = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU")
OUT = PROJECT / "baseline_callers"

def open_text(path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def vcf_to_csv(vcf_path, out_csv, caller_name, pass_only=True):
    rows = []

    with open_text(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0]
            pos = parts[1]
            ref = parts[3]
            alt = parts[4]
            filt = parts[6]

            if pass_only:
                if filt not in ["PASS", "."]:
                    continue

            # keep only SNVs for fair comparison
            for a in alt.split(","):
                if len(ref) == 1 and len(a) == 1:
                    rows.append({
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": a,
                        "filter": filt,
                        "caller": caller_name
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{caller_name}: saved {len(df)} PASS SNV calls to {out_csv}")

vcf_to_csv(
    OUT / "mutect2.vcf.gz",
    OUT / "mutect2_pass_snv_baseline.csv",
    "Mutect2",
    pass_only=True
)

# This Strelka file is demo only. Convert for inspection, but do not use in manuscript.
strelka_demo = OUT / "strelka2_somatic.snvs.vcf.gz"
if strelka_demo.exists():
    vcf_to_csv(
        strelka_demo,
        OUT / "strelka2_demo_pass_snv_baseline_DO_NOT_USE.csv",
        "Strelka2_demo",
        pass_only=True
    )