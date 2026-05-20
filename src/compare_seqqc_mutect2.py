from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU")

TABLES = PROJECT / "final_publication_results" / "tables"
DATA_ROOT = PROJECT / "data_root"
OUTDIR = PROJECT / "baseline_callers"

SEQQC_PRED = TABLES / "test_predictions_seqQC.csv"
SITES = DATA_ROOT / "sites.csv"
MUTECT2_CSV = OUTDIR / "mutect2_pass_snv_baseline.csv"

OUT_METRICS = OUTDIR / "seqqc_vs_mutect2_pass_metrics.csv"
OUT_PLOT = OUTDIR / "seqqc_vs_mutect2_pass_comparison.png"
OUT_MATCHED = OUTDIR / "seqqc_mutect2_pass_matched_test_loci.csv"


def normalize_chrom(x):
    x = str(x)
    if x.startswith("chr"):
        x = x[3:]
    return x


def make_variant_key(df):
    out = df.copy()
    out["chrom"] = out["chrom"].map(normalize_chrom)
    out["pos"] = pd.to_numeric(out["pos"], errors="coerce").astype("Int64")
    out["ref"] = out["ref"].astype(str).str.upper()
    out["alt"] = out["alt"].astype(str).str.upper().str.split(",").str[0]
    out["variant_key"] = (
        out["chrom"].astype(str)
        + ":"
        + out["pos"].astype(str)
        + ":"
        + out["ref"]
        + ":"
        + out["alt"]
    )
    return out


def compute_metrics(name, y_true, score, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    try:
        auroc = roc_auc_score(y_true, score)
    except Exception:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, score)
    except Exception:
        auprc = np.nan

    return {
        "Model": name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "n_test": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
    }


def main():
    print("Reading SeqQC predictions:", SEQQC_PRED)
    pred = pd.read_csv(SEQQC_PRED)

    print("Reading site coordinates:", SITES)
    sites = pd.read_csv(SITES)

    sites = sites.rename(columns={
        "Chromosome": "chrom",
        "Start_Position": "pos",
        "Reference_Allele": "ref",
        "Tumor_Seq_Allele2": "alt",
    })

    required_pred = ["row_index", "y_true", "prob", "pred"]
    missing = [c for c in required_pred if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing columns in SeqQC prediction file: {missing}")

    required_sites = ["chrom", "pos", "ref", "alt"]
    missing = [c for c in required_sites if c not in sites.columns]
    if missing:
        raise ValueError(f"Missing columns in sites.csv after renaming: {missing}")

    row_index = pred["row_index"].astype(int).to_numpy()

    test_loci = sites.iloc[row_index].reset_index(drop=True).copy()
    test_loci["row_index"] = row_index
    test_loci["y_true"] = pred["y_true"].astype(int).values
    test_loci["seqqc_prob"] = pred["prob"].astype(float).values
    test_loci["seqqc_pred"] = pred["pred"].astype(int).values

    test_loci = make_variant_key(test_loci)

    print("Test loci:", test_loci.shape)
    print("y_true counts:")
    print(test_loci["y_true"].value_counts())

    print("Reading Mutect2 PASS SNV baseline:", MUTECT2_CSV)
    mut = pd.read_csv(MUTECT2_CSV)

    if len(mut) == 0:
        raise ValueError("Mutect2 PASS SNV CSV is empty. Check VCF FILTER values.")

    mut = make_variant_key(mut)
    mut_keys = set(mut["variant_key"].dropna().astype(str).unique())

    test_loci["mutect2_pred"] = test_loci["variant_key"].astype(str).isin(mut_keys).astype(int)
    test_loci["mutect2_score"] = test_loci["mutect2_pred"].astype(float)

    y_true = test_loci["y_true"].astype(int).to_numpy()

    metrics = [
        compute_metrics(
            "SeqQC-Former",
            y_true,
            test_loci["seqqc_prob"].to_numpy(),
            test_loci["seqqc_pred"].to_numpy(),
        ),
        compute_metrics(
            "Mutect2 PASS SNVs",
            y_true,
            test_loci["mutect2_score"].to_numpy(),
            test_loci["mutect2_pred"].to_numpy(),
        ),
    ]

    metrics_df = pd.DataFrame(metrics).round(6)

    metrics_df.to_csv(OUT_METRICS, index=False)
    test_loci.to_csv(OUT_MATCHED, index=False)

    print("\nComparison metrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved:", OUT_METRICS)
    print("Saved:", OUT_MATCHED)

    plot_df = metrics_df[["Model", "Precision", "Recall", "F1", "AUPRC"]].copy()
    x = np.arange(len(plot_df))
    width = 0.2

    plt.figure(figsize=(9, 5))
    plt.bar(x - 1.5 * width, plot_df["Precision"], width, label="Precision")
    plt.bar(x - 0.5 * width, plot_df["Recall"], width, label="Recall")
    plt.bar(x + 0.5 * width, plot_df["F1"], width, label="F1")
    plt.bar(x + 1.5 * width, plot_df["AUPRC"], width, label="AUPRC")

    plt.xticks(x, plot_df["Model"], rotation=10)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric value")
    plt.title("SeqQC-Former versus Mutect2 on matched SEQC2 test loci")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_PLOT)


if __name__ == "__main__":
    main()