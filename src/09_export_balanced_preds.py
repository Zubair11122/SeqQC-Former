# export_balanced_preds.py
import os, sys, importlib.util, csv
from pathlib import Path
import yaml, h5py, numpy as np, torch

# config/paths
CFG_PATH = Path(__file__).with_name("config.yaml")
cfg = yaml.safe_load(open(CFG_PATH))
root = Path(cfg["data_root"]).expanduser()
h5   = root / "features.h5"
ckpt = root / "rep_error_net.ckpt"
assert h5.exists(), f"Missing HDF5: {h5}"
assert ckpt.exists(), f"Missing checkpoint: {ckpt}"

# import Net
train_py = Path(__file__).with_name("03_A_train_lightning.py")
spec = importlib.util.spec_from_file_location("tm", str(train_py))
tm = importlib.util.module_from_spec(spec)
sys.modules["tm"] = tm
spec.loader.exec_module(tm)
Net = tm.Net

device = torch.device(f"cuda:{cfg.get('gpu_id',0)}" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

# balanced indices
with h5py.File(h5, "r") as f:
    y_all = np.asarray(f["y"], dtype=int)
pos = np.where(y_all == 1)[0]
neg = np.where(y_all == 0)[0]
rng = np.random.default_rng(42)
neg = rng.choice(neg, size=len(pos), replace=True)
idx = np.concatenate([pos, neg]); rng.shuffle(idx)
targets = y_all[idx]

# model
model = Net.load_from_checkpoint(str(ckpt), cfg=cfg, strict=False).eval().to(device)

# inference
B = 1024
probs = np.empty(len(idx), dtype=np.float32)
def _to_ch_first(x):  # (L,4)->(4,L) if needed
    return x.transpose(1,0) if (x.ndim==2 and x.shape[1]==4) else x

with h5py.File(h5, "r") as f:
    dseq, dqc = f["seq"], f["qc"]
    with torch.no_grad():
        for s in range(0, len(idx), B):
            b = idx[s:s+B]
            xseq = np.stack([_to_ch_first(dseq[i]) for i in b], 0)
            xqc  = np.stack([dqc[i]                for i in b], 0)
            p = torch.sigmoid(
                model(torch.tensor(xseq).float().to(device),
                      torch.tensor(xqc ).float().to(device))
            ).cpu().numpy().astype(np.float32)
            probs[s:s+len(b)] = p

# write CSV
out_csv = root / "balanced_preds.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "label", "prob", "pred_0.5"])
    for i, p in zip(idx, probs):
        w.writerow([int(i), int(y_all[i]), float(p), int(p >= 0.5)])

print(f"[INFO] saved: {out_csv}")
