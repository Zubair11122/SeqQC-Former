#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_ci_auc(y, p, n=200, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    vals=[]
    for _ in range(n):
        s = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True)
        ])
        vals.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(vals, [2.5,97.5])
    return float(lo), float(hi)
