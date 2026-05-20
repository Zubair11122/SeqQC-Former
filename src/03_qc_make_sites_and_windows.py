#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import yaml

CFG = Path("config.yaml")
cfg = yaml.safe_load(open(CFG))
root = Path(cfg["data_root"])

variants = pd.read_csv(root / "variants_labeled.csv")
sites = variants[["Chromosome","Start_Position","Reference_Allele","Tumor_Seq_Allele2"]].drop_duplicates()
sites.to_csv(root / "sites.csv",index=False)
print("Saved sites.csv")
