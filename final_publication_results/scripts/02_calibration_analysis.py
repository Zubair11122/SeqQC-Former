from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path("final_publication_results")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

pred_file = TABLES / "test_predictions_seqQC.csv"
df = pd.read_csv(pred_file)

print("Columns:", df.columns.tolist())

# Your file columns from final_numbers.log:
label_col = "y_true"
prob_col = "prob"

if label_col not in df.columns:
    raise ValueError(f"Missing label column '{label_col}'. Found: {df.columns.tolist()}")

if prob_col not in df.columns:
    raise ValueError(f"Missing probability column '{prob_col}'. Found: {df.columns.tolist()}")

y_true = df[label_col].astype(int).to_numpy()
y_prob = df[prob_col].astype(float).to_numpy()
y_prob = np.clip(y_prob, 0, 1)

# Main calibration metrics
brier = brier_score_loss(y_true, y_prob)

# Expected calibration error using 10 bins
bins = np.linspace(0, 1, 11)
bin_ids = np.digitize(y_prob, bins, right=True) - 1
bin_ids = np.clip(bin_ids, 0, 9)

ece = 0.0
rows = []

for i in range(10):
    mask = bin_ids == i
    if mask.sum() == 0:
        continue

    mean_prob = float(y_prob[mask].mean())
    observed = float(y_true[mask].mean())
    weight = float(mask.mean())
    gap = abs(observed - mean_prob)
    ece += weight * gap

    rows.append({
        "bin": i,
        "bin_start": float(bins[i]),
        "bin_end": float(bins[i + 1]),
        "n": int(mask.sum()),
        "mean_predicted_probability": mean_prob,
        "observed_positive_fraction": observed,
        "absolute_gap": gap,
        "bin_weight": weight
    })

cal_bins = pd.DataFrame(rows)
cal_bins.to_csv(TABLES / "calibration_bins.csv", index=False)

metrics = pd.DataFrame([
    {"metric": "Brier score", "value": float(brier)},
    {"metric": "Expected calibration error", "value": float(ece)},
    {"metric": "n_test", "value": int(len(y_true))},
    {"metric": "n_positive", "value": int((y_true == 1).sum())},
    {"metric": "n_negative", "value": int((y_true == 0).sum())},
    {"metric": "positive_fraction", "value": float(y_true.mean())},
    {"metric": "mean_predicted_probability", "value": float(y_prob.mean())},
    {"metric": "median_predicted_probability", "value": float(np.median(y_prob))}
])
metrics.to_csv(TABLES / "calibration_metrics.csv", index=False)

# Calibration curve
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="SeqQC-Former")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title("Calibration curve on held-out SEQC2 test set")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "calibration_curve.png", dpi=300)
plt.close()

# Probability histogram by true label
plt.figure(figsize=(7, 5))
plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Negative loci")
plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Positive loci")
plt.xlabel("Predicted probability")
plt.ylabel("Number of loci")
plt.title("Predicted probability distribution by true label")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "probability_by_label_histogram.png", dpi=300)
plt.close()

print("\nCalibration metrics:")
print(metrics.to_string(index=False))
print("\nCalibration bins:")
print(cal_bins.to_string(index=False))
print("\nSaved:")
print(TABLES / "calibration_metrics.csv")
print(TABLES / "calibration_bins.csv")
print(FIGS / "calibration_curve.png")
print(FIGS / "probability_by_label_histogram.png")
