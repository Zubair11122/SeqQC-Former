# save as: make_seqqc_clean.py
import pandas as pd
from pathlib import Path
import re

# --- paths (edit if your tree is different) ---
DATA_ROOT = Path(r"C:/Users/Zubair/Desktop/Project/data_root")  # or /home/BD-4/rep_data on server
SEQQC_IN  = DATA_ROOT / "full_preds_with_keys.csv"
TRUTH_CSV = DATA_ROOT / "variants_labeled.csv"
SEQQC_OUT = DATA_ROOT / "full_preds_clean_by_key.csv"

# --- load SeqQC (must contain 'key' and 'prob') ---
s = pd.read_csv(SEQQC_IN)
if 'key' not in s.columns:
    raise SystemExit("full_preds_with_keys.csv must have a 'key' column.")
if 'prob' not in s.columns:
    raise SystemExit("full_preds_with_keys.csv must have a 'prob' column.")
# collapse duplicates by key: pick max prob per key
s = s.groupby('key', as_index=False)['prob'].max()

# --- load truth and normalize to build matching 'key' ---
t = pd.read_csv(TRUTH_CSV)
ren = {}
for c in t.columns:
    lc = c.lower()
    if lc in ("chromosome","chrom","#chrom","chr"): ren[c] = "chrom"
    elif lc in ("start_position","pos","position","start"): ren[c] = "pos"
    elif lc in ("reference_allele","ref","reference"): ren[c] = "ref"
    elif lc in ("tumor_seq_allele2","alt","alt_allele"): ren[c] = "alt"
    elif lc in ("seqc2_positive","label","y"): ren[c] = "label"
t = t.rename(columns=ren)
need = {"chrom","pos","ref","alt","label"}
miss = need - set(t.columns)
if miss:
    raise SystemExit(f"variants_labeled.csv missing columns: {miss}")

def norm_allele(x): return str(x).split(',')[0].split(';')[0].upper()
def norm_pos(x):
    x = str(x)
    return str(int(float(x))) if re.match(r'^\d+(\.\d+)?$', x) else x

t['chrom'] = t['chrom'].astype(str).str.replace(r'^chr','', regex=True)
t['pos']   = t['pos'].apply(norm_pos)
t['ref']   = t['ref'].apply(norm_allele)
t['alt']   = t['alt'].apply(norm_allele)
t['key']   = t['chrom'] + ":" + t['pos'] + ":" + t['ref'] + ":" + t['alt']
t['label'] = t['label'].astype(int)
t = t[['key','label']].drop_duplicates('key')

# --- join: trust truth labels ---
m = s.merge(t, on='key', how='inner')
n_before = len(s); n_after = len(m)
print(f"[INFO] SeqQC keys before: {n_before}  after merge w/ truth: {n_after}  (dropped {n_before-n_after})")

pos = int((m['label']==1).sum()); neg = int((m['label']==0).sum())
print(f"[INFO] Class balance after cleaning: pos={pos}  neg={neg}")

m.to_csv(SEQQC_OUT, index=False)
print(f"[OK] Wrote {SEQQC_OUT}")
