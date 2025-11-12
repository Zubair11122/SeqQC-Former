# eval_balanced.py — balanced evaluation without OOM (h5py fixed)

import os, sys, importlib.util
from pathlib import Path
import yaml, h5py, numpy as np, torch

# ---------- Config & paths ----------
CFG_PATH = Path(__file__).with_name("config.yaml")
cfg = yaml.safe_load(open(CFG_PATH))
data_root = Path(cfg["data_root"]).expanduser()
h5_path   = data_root / "features.h5"
ckpt_path = data_root / "rep_error_net.ckpt"
assert h5_path.exists(), f"Missing HDF5: {h5_path}"
assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

# ---------- Import Net from filename starting with number ----------
train_file = Path(__file__).with_name("03_A_train_lightning.py")
spec = importlib.util.spec_from_file_location("trainmod", str(train_file))
trainmod = importlib.util.module_from_spec(spec)
sys.modules["trainmod"] = trainmod
spec.loader.exec_module(trainmod)
Net = trainmod.Net

# ---------- Optional: sklearn ----------
try:
    from sklearn.metrics import classification_report, roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False
    from torchmetrics.classification import BinaryAUROC

# ---------- Device ----------
device = torch.device(f"cuda:{cfg.get('gpu_id',0)}" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------- Build balanced index set ----------
with h5py.File(str(h5_path), "r") as f:
    y_all = np.asarray(f["y"], dtype=np.int64)
pos = np.where(y_all == 1)[0]
neg = np.where(y_all == 0)[0]
if len(pos) == 0 or len(neg) == 0:
    raise RuntimeError("Cannot balance: need both positives and negatives in y.")
rng = np.random.default_rng(42)
neg_bal = rng.choice(neg, size=len(pos), replace=True)
idx = np.concatenate([pos, neg_bal])
rng.shuffle(idx)
print(f"[INFO] Balanced set size: {len(idx)} (pos={len(pos)}, neg={len(pos)})")

# ---------- Load model ----------
print(f"[INFO] Loading checkpoint: {ckpt_path}")
model = Net.load_from_checkpoint(str(ckpt_path), cfg=cfg, strict=False)
model.eval().to(device)

# ---------- Batched inference (h5py-friendly indexing) ----------
B = 1024
probs = np.empty(len(idx), dtype=np.float32)
targets = y_all[idx].astype(np.int64)

def _to_ch_first(x):
    # ensure (4, L) for Conv1d
    if x.ndim == 2 and x.shape[0] != 4 and x.shape[1] == 4:
        return np.transpose(x, (1, 0))  # (L,4)->(4,L)
    return x

with h5py.File(str(h5_path), "r") as f:
    dseq, dqc = f["seq"], f["qc"]
    with torch.no_grad():
        for s in range(0, len(idx), B):
            e = min(s + B, len(idx))
            batch_idx = idx[s:e]

            # h5py cannot take shuffled arrays directly; use per-index reads
            xseq = np.stack([_to_ch_first(dseq[i]) for i in batch_idx], axis=0)
            xqc  = np.stack([dqc[i]               for i in batch_idx], axis=0)

            t_seq = torch.tensor(xseq, dtype=torch.float32, device=device)
            t_qc  = torch.tensor(xqc,  dtype=torch.float32, device=device)

            logits = model(t_seq, t_qc)
            probs[s:e] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

# ---------- Metrics ----------
threshold = 0.5
pred = (probs >= threshold).astype(np.int64)

if HAVE_SK:
    auroc = roc_auc_score(targets, probs)
    print(f"\nBalanced AUROC: {auroc:.3f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(targets, pred, digits=3))
else:
    print("[WARN] scikit-learn not available; using TorchMetrics + manual report.\n")
    from torchmetrics.classification import BinaryAUROC
    auroc = BinaryAUROC()(torch.tensor(probs), torch.tensor(targets)).item()
    print(f"Balanced AUROC: {auroc:.3f}")
    tp = int(((pred == 1) & (targets == 1)).sum())
    tn = int(((pred == 0) & (targets == 0)).sum())
    fp = int(((pred == 1) & (targets == 0)).sum())
    fn = int(((pred == 0) & (targets == 1)).sum())
    def _prf(tp, fp, fn):
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        return p, r, f
    p_pos, r_pos, f1_pos = _prf(tp, fp, fn)
    p_neg, r_neg, f1_neg = _prf(tn, fn, fp)
    acc = (tp + tn) / max(1, len(targets))
    print(f"\nManual classification report (threshold=0.5)")
    print(f"  Pos: precision={p_pos:.3f} recall={r_pos:.3f} f1={f1_pos:.3f}")
    print(f"  Neg: precision={p_neg:.3f} recall={r_neg:.3f} f1={f1_neg:.3f}")
    print(f"  Acc: {acc:.3f}")
