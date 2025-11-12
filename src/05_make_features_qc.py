#!/usr/bin/env python3
import yaml, h5py, numpy as np, pandas as pd, pyfaidx
from pathlib import Path
from tqdm import tqdm

cfg = yaml.safe_load(open("config.yaml"))
root = Path(cfg["data_root"])
fa   = pyfaidx.Fasta(cfg["ref_fasta"])
W    = int(cfg["window_bp"])

df = pd.read_pickle(root/"variants_labeled.pkl")
df = df.rename(columns={"Chromosome":"chrom","Start_Position":"pos","Reference_Allele":"ref","Tumor_Seq_Allele2":"alt","seqc2_positive":"label"})
df["chrom"] = df["chrom"].astype(str).str.replace("^chr","",regex=True)
df["key"]   = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]

qc = pd.read_csv(root/"qc_merged.csv")
df = df.merge(qc, on="key", how="left").fillna(0.0)

N  = len(df); L = 2*W+1
with h5py.File(root/"features.h5","w") as h5:
    dseq = h5.create_dataset("seq", (N,4,L), "u1")
    dqc  = h5.create_dataset("qc",  (N,7),   "f4")   # DP, AD, VAF, MQ, SB, MAP, RTIM
    dy   = h5.create_dataset("y",   (N,),    "u1")

    BASE = {b: np.eye(4,dtype="u1")[i] for i,b in enumerate("ACGT")}
    def onehot(s):
        arr = np.zeros((4,len(s)), "u1")
        for i,ch in enumerate(s.upper()):
            if ch in BASE: arr[:,i]=BASE[ch]
        return arr

    for i,r in tqdm(df.iterrows(), total=N):
        k = "chr"+str(r.chrom)
        seq = fa[k][int(r.pos)-1-W : int(r.pos)+W].seq
        dseq[i] = onehot(seq)
        dqc[i]  = [r.DP, r.AD, r.VAF, r.MQ, r.SB, r.MAP, r.RTIM]
        dy[i]   = int(r.label)

print("Wrote:", root/"features.h5")
