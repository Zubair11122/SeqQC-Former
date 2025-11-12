#!/usr/bin/env python3
from pathlib import Path
import argparse, pysam, pandas as pd

def pileup_qc(bam, chrom, pos1, alt):
    dp=ad=0; mq_sum=bq_sum=0; fwd=rev=0
    sam = pysam.AlignmentFile(bam,"rb")
    for col in sam.pileup(chrom, pos1-1, pos1, truncate=True, stepper="samtools", max_depth=100000):
        if col.pos != pos1-1: continue
        dp = col.nsegments
        for pr in col.pileups:
            if pr.is_del or pr.is_refskip: continue
            b = pr.alignment.query_sequence[pr.query_position]
            q = pr.alignment.query_qualities[pr.query_position]
            mq= pr.alignment.mapping_quality
            if b and b.upper()==alt.upper():
                ad += 1; bq_sum += (q or 0); mq_sum += mq
                if pr.alignment.is_reverse: rev += 1
                else: fwd += 1
    vaf = ad/dp if dp else 0.0
    mq  = mq_sum/max(1,ad)
    sb  = abs(fwd-rev)/max(1,ad)
    return dp, ad, vaf, mq, sb

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)      # data_root/sites.csv
    ap.add_argument("--tumor_bam", required=True)
    ap.add_argument("--out", required=True)        # data_root/qc_bam.csv
    args=ap.parse_args()

    sites = pd.read_csv(args.sites)
    rows=[]
    for r in sites.itertuples(index=False):
        chrom=str(r.chrom).lstrip("chr")
        dp,ad,vaf,mq,sb = pileup_qc(args.tumor_bam, chrom, int(r.pos), r.alt)
        key = f"{chrom}:{int(r.pos)}:{r.ref}:{r.alt}"
        rows.append(dict(key=key, DP=dp, AD=ad, VAF=vaf, MQ=mq, SB=sb))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote:", args.out)
