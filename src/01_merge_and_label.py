#!/usr/bin/env python3
from pathlib import Path
import yaml, gzip, glob, pandas as pd, sys

CFG = Path("/home/zubair/Project/config.yaml")  # Directly use full path
cfg = yaml.safe_load(open(CFG))
root = Path(cfg["data_root"]).resolve()
out_pkl = root / "variants_labeled.pkl"

def open_text(p):
    with open(p, "rb") as bf:
        gz = bf.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rt") if gz else open(p, "r", encoding="utf-8", errors="replace")

# Negatives from GBM MAFs (GRCh38)
required = ["chromosome","start_position","reference_allele","tumor_seq_allele2","ncbi_build"]
dfs=[]
for maf in glob.glob(str(root/"gbm_mafs/*.maf*")):
    with open_text(maf) as fh:
        header=None
        for line in fh:
            if line.startswith("#"): continue
            header=[h.strip() for h in line.rstrip("\n").split("\t")]; break
        if not header: continue
        hl=[h.lower() for h in header]
        if any(c not in hl for c in required):
            print(f"Missing required cols in {maf}", file=sys.stderr); sys.exit(1)
        idx={c:hl.index(c) for c in required}
        rows=[]
        for line in fh:
            if line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            if f[idx["ncbi_build"]]!="GRCh38": continue
            rows.append([f[idx["chromosome"]], int(f[idx["start_position"]]),
                         f[idx["reference_allele"]], f[idx["tumor_seq_allele2"]], 0])
        if rows:
            dfs.append(pd.DataFrame(rows, columns=["Chromosome","Start_Position","Reference_Allele","Tumor_Seq_Allele2","seqc2_positive"]))

# Positives from SEQC2 truth VCFs (SNV + INDEL)
pos=[]
for vcf in glob.glob(str(root/"seqc2_truth/*.vcf*")):
    with open_text(vcf) as fh:
        for line in fh:
            if line.startswith("#"): continue
            c=line.rstrip("\n").split("\t")
            chrom=c[0].lstrip("chr"); posi=int(c[1]); ref=c[3]
            for alt in c[4].split(","):
                pos.append([chrom,posi,ref,alt,1])
dfpos = pd.DataFrame(pos, columns=["Chromosome","Start_Position","Reference_Allele","Tumor_Seq_Allele2","seqc2_positive"])

# Merge & save
all_dfs = dfs + [dfpos]
if not all_dfs: raise SystemExit("No data found in gbm_mafs/ and seqc2_truth/")
pd.concat(all_dfs, ignore_index=True).to_pickle(out_pkl)
print("Saved →", out_pkl)
