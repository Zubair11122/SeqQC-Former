# analyze_thresholds.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

BASE = Path(r"C:/Users/Zubair/Desktop/rep_error_project/rep_data")
CSV_PATH = BASE / "full_preds.csv"
SWEEP_CSV = BASE / "threshold_sweep.csv"
FIG_NORM_CM = BASE / "fig_confusion_normalized.png"

BASE.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV_PATH)
y = pd.to_numeric(df["label"])
p = pd.to_numeric(df["prob"])

# Build a good set of candidate thresholds (unique probs + regular grid)
cand = np.unique(np.concatenate([p.values, np.linspace(0, 1, 201)]))
rows = []
for t in cand:
    pred = (p >= t).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    acc  = accuracy_score(y, pred)
    tpr  = tp / (tp + fn) if (tp + fn) else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    youden = tpr - fpr
    rows.append(dict(threshold=t, tp=tp, fp=fp, tn=tn, fn=fn,
                     precision=prec, recall=rec, f1=f1, accuracy=acc, tpr=tpr, fpr=fpr, youden=youden))

sweep = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
sweep.to_csv(SWEEP_CSV, index=False)

# Pick best-by-F1 and best-by-Youden
best_f1 = sweep.loc[sweep["f1"].idxmax()]
best_yj = sweep.loc[sweep["youden"].idxmax()]
print(f"Best F1 threshold: {best_f1.threshold:.4f}  F1={best_f1.f1:.4f}  "
      f"Prec={best_f1.precision:.4f}  Rec={best_f1.recall:.4f}  Acc={best_f1.accuracy:.4f}")
print(f"Best Youden threshold: {best_yj.threshold:.4f}  J={best_yj.youden:.4f}  "
      f"TPR={best_yj.tpr:.4f}  FPR={best_yj.fpr:.4f}")

# Normalized confusion matrix at the best-F1 threshold
t = float(best_f1.threshold)
pred = (p >= t).astype(int)
cm = confusion_matrix(y, pred, labels=[0,1], normalize='true')  # row-normalized
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
plt.title(f"Normalized Confusion Matrix @ threshold={t:.3f} (best F1)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG_NORM_CM, dpi=900, bbox_inches="tight")
plt.close()

print(f"Saved sweep CSV: {SWEEP_CSV}")
print(f"Saved normalized CM: {FIG_NORM_CM}")
