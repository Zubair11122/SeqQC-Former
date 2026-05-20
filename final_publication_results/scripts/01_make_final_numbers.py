from pathlib import Path
import pandas as pd
import h5py
import numpy as np

ROOT = Path("final_publication_results")
TABLES = ROOT / "tables"

rows = []

def add(category, item, value):
    rows.append({"Category": category, "Item": item, "Value": value})

# -------------------------
# HDF5 summary
# -------------------------
features_h5 = ROOT / "features.h5"

if features_h5.exists():
    with h5py.File(features_h5, "r") as f:
        print("HDF5 keys:", list(f.keys()))
        y = f["labels"][:]

        add("SEQC2 dataset", "Total loci", len(y))
        add("SEQC2 dataset", "Positive labels", int((y == 1).sum()))
        add("SEQC2 dataset", "Negative labels", int((y == 0).sum()))
        add("SEQC2 dataset", "Positive fraction", float((y == 1).mean()))

        if "sequences" in f:
            add("Feature matrix", "Sequences shape", str(f["sequences"].shape))
        if "qc_features" in f:
            add("Feature matrix", "QC features shape", str(f["qc_features"].shape))

        # Try to read QC names if stored as attributes
        for attr_name in ["qc_column_names", "qc_columns", "feature_names"]:
            if attr_name in f.attrs:
                try:
                    names = f.attrs[attr_name]
                    names = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in names
                    ]
                    add("Feature matrix", "QC feature names", "; ".join(names))
                except Exception:
                    add("Feature matrix", "QC feature names", str(f.attrs[attr_name]))
else:
    add("Missing", "features.h5", "not found")

# -------------------------
# Split summary
# -------------------------
splits_file = TABLES / "splits.csv"

if splits_file.exists():
    splits = pd.read_csv(splits_file)

    add("Split", "Split file rows", len(splits))

    for split_name in ["train", "val", "test"]:
        n = int((splits["split"] == split_name).sum())
        add("Split", f"{split_name} samples", n)

    # If labels available, also count class distribution per split
    if features_h5.exists():
        with h5py.File(features_h5, "r") as f:
            y = f["labels"][:]
        for split_name in ["train", "val", "test"]:
            idx = splits.loc[splits["split"] == split_name, "index"].values
            add("Split class balance", f"{split_name} positives", int((y[idx] == 1).sum()))
            add("Split class balance", f"{split_name} negatives", int((y[idx] == 0).sum()))
            add("Split class balance", f"{split_name} positive fraction", float((y[idx] == 1).mean()))
else:
    add("Missing", "splits.csv", "not found")

# -------------------------
# Test metrics
# -------------------------
metrics_file = TABLES / "test_metrics_seqQC.csv"

if metrics_file.exists():
    metrics = pd.read_csv(metrics_file)
    add("SEQC2 test performance", "Metrics file rows", len(metrics))

    if len(metrics) > 0:
        for col in metrics.columns:
            add("SEQC2 test performance", col, metrics[col].iloc[0])
else:
    add("Missing", "test_metrics_seqQC.csv", "not found")

# -------------------------
# Threshold
# -------------------------
threshold_file = TABLES / "best_threshold.txt"

if threshold_file.exists():
    add("Threshold", "Best threshold", threshold_file.read_text().strip().splitlines()[0])
else:
    add("Missing", "best_threshold.txt", "not found")

# -------------------------
# Test predictions summary
# -------------------------
pred_file = TABLES / "test_predictions_seqQC.csv"

if pred_file.exists():
    pred = pd.read_csv(pred_file)
    add("SEQC2 test predictions", "Rows", len(pred))
    add("SEQC2 test predictions", "Columns", "; ".join(pred.columns))
else:
    add("Missing", "test_predictions_seqQC.csv", "not found")

# -------------------------
# GBM main predictions
# -------------------------
gbm_file = TABLES / "gbm_predictions.csv"

if gbm_file.exists():
    gbm = pd.read_csv(gbm_file)
    add("GBM main", "Total SNVs", len(gbm))
    add("GBM main", "Columns", "; ".join(gbm.columns))

    if "rep_error_probability" in gbm.columns:
        add("GBM main", "Mean probability", float(gbm["rep_error_probability"].mean()))
        add("GBM main", "Median probability", float(gbm["rep_error_probability"].median()))
        add("GBM main", "Min probability", float(gbm["rep_error_probability"].min()))
        add("GBM main", "Max probability", float(gbm["rep_error_probability"].max()))

    if "rep_error_predicted" in gbm.columns:
        n_art = int(gbm["rep_error_predicted"].sum())
        add("GBM main", "Predicted artifact candidates", n_art)
        add("GBM main", "Artifact candidate fraction", n_art / len(gbm))
        add("GBM main", "Retained variants", len(gbm) - n_art)
else:
    add("Missing", "gbm_predictions.csv", "not found")

# -------------------------
# GBM summary publication table
# -------------------------
gbm_summary_file = TABLES / "gbm_summary_for_publication.csv"

if gbm_summary_file.exists():
    summary = pd.read_csv(gbm_summary_file)
    add("GBM publication summary", "Rows", len(summary))
    add("GBM publication summary", "Columns", "; ".join(summary.columns))
else:
    add("Missing", "gbm_summary_for_publication.csv", "not found")

# -------------------------
# GBM gene summary
# -------------------------
gene_file = TABLES / "gbm_gene_level_artifact_summary.csv"

if gene_file.exists():
    gene = pd.read_csv(gene_file)
    add("GBM gene summary", "Rows", len(gene))
    add("GBM gene summary", "Columns", "; ".join(gene.columns))
else:
    add("Missing", "gbm_gene_level_artifact_summary.csv", "not found")

# -------------------------
# GBM clip3 sensitivity
# -------------------------
clip3_file = TABLES / "gbm_predictions_clip3.csv"

if clip3_file.exists():
    clip3 = pd.read_csv(clip3_file)
    add("GBM clip3 sensitivity", "Total SNVs", len(clip3))

    if "rep_error_probability" in clip3.columns:
        add("GBM clip3 sensitivity", "Mean probability", float(clip3["rep_error_probability"].mean()))
        add("GBM clip3 sensitivity", "Median probability", float(clip3["rep_error_probability"].median()))

    if "rep_error_predicted" in clip3.columns:
        n_art = int(clip3["rep_error_predicted"].sum())
        add("GBM clip3 sensitivity", "Predicted artifact candidates", n_art)
        add("GBM clip3 sensitivity", "Artifact candidate fraction", n_art / len(clip3))
        add("GBM clip3 sensitivity", "Retained variants", len(clip3) - n_art)
else:
    add("Missing", "gbm_predictions_clip3.csv", "not found")

# -------------------------
# Save output
# -------------------------
out = TABLES / "final_manuscript_numbers.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)

print("\nFinal manuscript numbers:")
print(df.to_string(index=False))
print("\nSaved:", out)