import re, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# --- read config + paths ---
CFG = Path(r"C:/Users/Zubair/Desktop/Project/config.yaml")
cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
data_root = Path(cfg["data_root"]).expanduser()
preds_csv = data_root / "full_preds.csv"               # from eval_full.py
map_csv   = data_root / "variants_labeled.csv"         # mapping with chrom/pos/ref/alt
out_csv   = data_root / "full_preds_with_keys.csv"

print(f"[INFO] preds: {preds_csv}")
print(f"[INFO] map  : {map_csv}")

# --- load preds ---
seqqc = pd.read_csv(preds_csv)
if list(seqqc.columns[:4]) != ["index","label","prob","pred_0.5"]:
    sys.exit(f"[ERROR] Unexpected columns in {preds_csv}: {seqqc.columns.tolist()}")
N = len(seqqc)
print(f"[INFO] N (pred rows) = {N}")

# --- load mapping (must have chrom/pos/ref/alt under flexible names) ---
raw = pd.read_csv(map_csv)

def pick(df, cands):
    norm = lambda s: re.sub(r'[^a-z0-9]+','', s.lower())
    have = {norm(c): c for c in df.columns}
    for x in cands:
        if norm(x) in have:
            return have[norm(x)]
    for c in df.columns:
        if any(norm(x) in norm(c) for x in cands):
            return c
    return None

chrom = pick(raw, ['chrom','chromosome','chr','#chrom','#CHROM','Chromosome','CHROM','contig','seqnames'])
pos   = pick(raw, ['pos','position','start','start_position','bp','start_bp','POS','Start_Position'])
ref   = pick(raw, ['ref','reference','ref_allele','reference_allele','REF','Reference_Allele'])
alt   = pick(raw, ['alt','alt_allele','alternate','ALT','Tumor_Seq_Allele2','Alt'])
if not all([chrom,pos,ref,alt]):
    sys.exit(f"[ERROR] {map_csv.name} missing one of chrom/pos/ref/alt; got: {list(raw.columns)[:16]}")

m = pd.DataFrame({
    "chrom": raw[chrom].astype(str).str.replace("^chr","",regex=True),
    "pos":   raw[pos].astype(str),
    "ref":   raw[ref].astype(str),
    "alt":   raw[alt].astype(str),
})
# normalize alleles / pos
m["ref"] = m["ref"].str.split("[,;]").str[0].str.upper()
m["alt"] = m["alt"].str.split("[,;]").str[0].str.upper()
m["pos"] = m["pos"].apply(lambda x: str(int(float(x))) if re.match(r"^\d+(\.\d+)?$", str(x)) else str(x))

L = len(m)
print(f"[INFO] L (mapping rows) = {L}")

# --- robust alignment: repeat enough times, then truncate to N ---
if L == 0:
    sys.exit("[ERROR] Mapping CSV is empty.")
if L != N:
    rep = math.ceil(N / L)
    print(f"[INFO] Repeating mapping x{rep} and truncating to N")
    m = pd.concat([m] * rep, ignore_index=True).iloc[:N].reset_index(drop=True)

assert len(m) == N, f"After alignment, mapping rows {len(m)} != N {N}"

# --- build keys + write output ---
key = m["chrom"] + ":" + m["pos"] + ":" + m["ref"] + ":" + m["alt"]
out = seqqc.copy()
# if the original preds don't have 'index' as a column (rare), create it
if "index" not in out.columns:
    out["index"] = np.arange(len(out))
out.insert(0, "key", key.values)
out = out[["key","label","prob","pred_0.5","index"]]
out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"[OK] Wrote {out_csv}")
