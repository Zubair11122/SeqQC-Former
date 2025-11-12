# eval_full.py — evaluate on the ENTIRE dataset (no balancing)
# Outputs:
#   - Prints AUROC and Average Precision over all samples
#   - Saves metrics to <data_root>/eval_full_metrics.txt
#   - Saves per-sample predictions to <data_root>/full_preds.csv

import os, sys, importlib.util
from pathlib import Path
import yaml, h5py, numpy as np, torch

# -------- Config & paths --------
CFG_PATH = Path(__file__).with_name("config.yaml")
cfg = yaml.safe_load(open(CFG_PATH))
data_root = Path(cfg["data_root"]).expanduser()
h5_path   = data_root / "features.h5"
ckpt_path = data_root / "rep_error_net.ckpt"

assert h5_path.exists(), f"Missing HDF5: {h5_path}"
assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

# -------- Import Net from 03_A_train_lightning.py --------
train_file = Path(__file__).with_name("03_A_train_lightning.py")
spec = importlib.util.spec_from_file_location("trainmod", str(train_file))
trainmod = importlib.util.module_from_spec(spec)
sys.modules["trainmod"] = trainmod
spec.loader.exec_module(trainmod)
Net = trainmod.Net

# -------- Optional sklearn (pretty metrics) --------
HAVE_SK = True
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
except Exception:
    HAVE_SK = False
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

# -------- Device & batch size --------
device = torch.device(f"cuda:{cfg.get('gpu_id',0)}" if torch.cuda.is_available() else "cpu")
B = 1024  # eval batch size; raise if you have plenty of VRAM
print(f"[INFO] Using device: {device}")
print(f"[INFO] Batch size: {B}")

# -------- Read labels and dataset size --------
with h5py.File(str(h5_path), "r") as f:
    y_all = np.asarray(f["y"], dtype=np.int64)
N = len(y_all)
print(f"[INFO] Total samples: {N}")

# -------- Load model --------
print(f"[INFO] Loading checkpoint: {ckpt_path}")
model = Net.load_from_checkpoint(str(ckpt_path), cfg=cfg, strict=False)
model.eval().to(device)

# -------- Inference over the full dataset --------
probs = np.empty(N, dtype=np.float32)

def ensure_channels_first_batch(x):
    """
    x: numpy array with shape (B, 4, L)  OR (B, L, 4)
    Returns x with shape (B, 4, L)
    """
    if x.ndim == 3 and x.shape[1] != 4 and x.shape[2] == 4:
        return np.transpose(x, (0, 2, 1))
    return x

with h5py.File(str(h5_path), "r") as f:
    dseq, dqc = f["seq"], f["qc"]

    with torch.no_grad():
        for s in range(0, N, B):
            e = min(s + B, N)
            # contiguous slices are HDF5-friendly
            xseq = dseq[s:e]         # shape (B, 4, L) or (B, L, 4)
            xqc  = dqc[s:e]          # shape (B, 7)

            xseq = ensure_channels_first_batch(xseq)

            t_seq = torch.tensor(xseq, dtype=torch.float32, device=device)
            t_qc  = torch.tensor(xqc,  dtype=torch.float32, device=device)

            logits = model(t_seq, t_qc)
            probs[s:e] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

# -------- Metrics --------
if HAVE_SK:
    auroc = float(roc_auc_score(y_all, probs))
    ap    = float(average_precision_score(y_all, probs))
else:
    auroc = float(BinaryAUROC()(torch.tensor(probs), torch.tensor(y_all)).item())
    ap    = float(BinaryAveragePrecision()(torch.tensor(probs), torch.tensor(y_all)).item())

print(f"\n[RESULT] Full-dataset AUROC: {auroc:.4f}")
print(f"[RESULT] Full-dataset AP   : {ap:.4f}")

# Optional quick report at 0.5 threshold
thr = 0.5
pred = (probs >= thr).astype(np.int64)
tp = int(((pred==1)&(y_all==1)).sum())
tn = int(((pred==0)&(y_all==0)).sum())
fp = int(((pred==1)&(y_all==0)).sum())
fn = int(((pred==0)&(y_all==1)).sum())
acc = (tp + tn) / max(1, N)
print(f"[INFO] Accuracy @0.5: {acc:.4f}  (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
if HAVE_SK:
    print("\n[INFO] Classification report @0.5:")
    print(classification_report(y_all, pred, digits=3))

# -------- Save artifacts --------
metrics_txt = data_root / "eval_full_metrics.txt"
metrics_txt.write_text(
    f"AUROC={auroc:.6f}\nAP={ap:.6f}\nACC@0.5={acc:.6f}\nTP={tp}\nTN={tn}\nFP={fp}\nFN={fn}\n"
)
print(f"[INFO] Saved metrics to: {metrics_txt}")

csv_path = data_root / "full_preds.csv"
with open(csv_path, "w") as f:
    f.write("index,label,prob,pred_0.5\n")
    for i, p in enumerate(probs):
        f.write(f"{i},{int(y_all[i])},{float(p):.6f},{int(p>=0.5)}\n")
print(f"[INFO] Saved per-sample predictions to: {csv_path}")
