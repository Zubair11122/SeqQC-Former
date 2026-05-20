from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path("final_publication_results")
TABLES = ROOT / "tables"
FIGS = ROOT / "publication_figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


def savefig(name):
    png = FIGS / f"{name}.png"
    pdf = FIGS / f"{name}.pdf"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", png)
    print("Saved:", pdf)


def box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.0, fontsize=8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.2,
        color="black"
    )
    ax.add_patch(arr)


# -----------------------------
# Figure 1: workflow + architecture
# FIXED: panel C moved upward, no boxes outside the canvas
# -----------------------------
def make_figure1():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "SeqQC-Former workflow and model architecture",
        ha="center", va="center", fontsize=15, weight="bold"
    )

    # Panel A
    ax.text(0.03, 0.90, "A. Dataset and feature construction", fontsize=11, weight="bold")

    y = 0.76
    w = 0.145
    h = 0.12
    xs = [0.03, 0.205, 0.38, 0.555, 0.73]

    steps = [
        "SEQC2-derived\ncandidate loci\n\n1,378 positive\n88,069 negative",
        "Reference sequence\nextraction\n\n501 bp window\n4 × 501 one-hot",
        "Tumor-normal\nread-level QC\n\n11 structured\nQC covariates",
        "Feature assembly\n\nfeatures.h5\nsequences + QC\nlabels",
        "Stratified split\n\nTrain: 57,245\nVal: 14,312\nTest: 17,890",
    ]

    for i, txt in enumerate(steps):
        box(ax, xs[i], y, w, h, txt, fc="#f3f3f3", ec="#555555", fontsize=8)
        if i < len(xs) - 1:
            arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # Panel B
    ax.text(0.03, 0.58, "B. Model inputs", fontsize=11, weight="bold")

    box(
        ax, 0.05, 0.42, 0.22, 0.12,
        "Sequence input\n\nOne-hot encoded\n4 × 501 tensor",
        fc="#eaf4ff", ec="#4c78a8", fontsize=9
    )

    box(
        ax, 0.33, 0.42, 0.22, 0.12,
        "QC input\n\n11 per-locus\nread-level features",
        fc="#fff1e6", ec="#f28e2b", fontsize=9
    )

    # Panel C
    ax.text(0.60, 0.62, "C. SeqQC-Former architecture", fontsize=11, weight="bold")

    # Sequence branch
    x0 = 0.58
    y0 = 0.54
    bw = 0.17
    bh = 0.05
    step = 0.075

    seq_labels = [
        "Sequence\n4 × 501",
        "Conv1D\n4→32, k=7",
        "Conv1D\n32→64, k=5",
        "Conv1D\n64→64, k=3",
        "Adaptive pool\n50 positions",
        "Transformer encoder\n2 layers, 4 heads\nd_model=64",
        "Sequence embedding\n64",
    ]

    seq_y_positions = []
    for i, lab in enumerate(seq_labels):
        yy = y0 - i * step
        seq_y_positions.append(yy)
        box(ax, x0, yy, bw, bh, lab,
            fc="#eaf4ff" if i in [0, 6] else "#d9ecff",
            ec="#4c78a8", fontsize=8)
        if i < len(seq_labels) - 1:
            arrow(ax, x0 + bw/2, yy, x0 + bw/2, yy - 0.025)

    # QC branch
    x1 = 0.80
    qc_labels = [
        "QC features\n11",
        "MLP\n11→32",
        "MLP\n32→16",
        "QC embedding\n16",
    ]
    qc_y_positions = [0.54, 0.43, 0.32, 0.21]
    for i, (lab, yy) in enumerate(zip(qc_labels, qc_y_positions)):
        box(ax, x1, yy, 0.15, bh, lab,
            fc="#fff1e6" if i in [0, 3] else "#ffe2c6",
            ec="#f28e2b", fontsize=8)
        if i < len(qc_labels) - 1:
            arrow(ax, x1 + 0.075, yy, x1 + 0.075, yy - 0.035)

    # Fusion / classifier boxes kept inside canvas
    fx = 0.67
    fusion_y = [0.18, 0.10, 0.02]
    fusion_labels = [
        "Concatenate\n64 + 16 = 80",
        "Classifier MLP\n80→64→32→1",
        "Artifact-prioritization\nscore",
    ]
    fusion_colors = ["#edf7e6", "#edf7e6", "#d5f0c1"]

    for yy, lab, fc in zip(fusion_y, fusion_labels, fusion_colors):
        box(ax, fx, yy, 0.21, 0.055, lab, fc=fc, ec="#59a14f", fontsize=8)

    # Branches to fusion
    arrow(ax, x0 + bw/2, seq_y_positions[-1], fx + 0.06, fusion_y[0] + 0.055)
    arrow(ax, x1 + 0.075, qc_y_positions[-1], fx + 0.15, fusion_y[0] + 0.055)
    arrow(ax, fx + 0.105, fusion_y[0], fx + 0.105, fusion_y[1] + 0.055)
    arrow(ax, fx + 0.105, fusion_y[1], fx + 0.105, fusion_y[2] + 0.055)

    plt.tight_layout()
    savefig("Figure_1_workflow_architecture")


# -----------------------------
# Figure 2: ROC, PR, confusion, probability histogram
# FIXED: histogram uses log-scale y-axis
# -----------------------------
def make_figure2():
    pred = pd.read_csv(TABLES / "test_predictions_seqQC.csv")
    y = pred["y_true"].astype(int).to_numpy()
    p = pred["prob"].astype(float).to_numpy()
    yhat = pred["pred"].astype(int).to_numpy()

    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y, p)
    cm = confusion_matrix(y, yhat)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {roc_auc:.6f}")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    # PR
    ax = axes[0, 1]
    ax.plot(recall, precision, lw=2, label=f"AUPRC = {auprc:.6f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")

    # Confusion matrix
    ax = axes[1, 0]
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix at threshold 0.75")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Histogram with log y-axis
    ax = axes[1, 1]
    ax.hist(p[y == 0], bins=50, alpha=0.7, label="Negative loci")
    ax.hist(p[y == 1], bins=50, alpha=0.7, label="Positive loci")
    ax.axvline(0.75, linestyle="--", lw=1.5, label="Threshold = 0.75")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted artifact-prioritization score")
    ax.set_ylabel("Number of loci (log scale)")
    ax.set_title("Score distribution by label")
    ax.legend()

    plt.tight_layout()
    savefig("Figure_2_test_performance")


# -----------------------------
# Figure 3: calibration + robustness
# -----------------------------
def make_figure3():
    bins = pd.read_csv(TABLES / "calibration_bins.csv")
    rob = pd.read_csv(TABLES / "robustness_test_only.csv")

    noise_col = "noise_sigma" if "noise_sigma" in rob.columns else rob.columns[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Calibration
    ax = axes[0]
    ax.plot(
        bins["mean_predicted_probability"],
        bins["observed_positive_fraction"],
        marker="o",
        lw=2,
        label="SeqQC-Former"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve")
    ax.legend()

    # Robustness
    ax = axes[1]
    ax.plot(rob[noise_col], rob["AUROC"], marker="o", label="AUROC")
    ax.plot(rob[noise_col], rob["AUPRC"], marker="s", label="AUPRC")
    ax.plot(rob[noise_col], rob["f1"], marker="^", label="F1")
    ax.set_xlabel("Gaussian noise σ added to scaled QC features")
    ax.set_ylabel("Metric value")
    ax.set_title("Robustness to QC-feature perturbation")
    ax.set_ylim(0.85, 1.01)
    ax.legend()

    plt.tight_layout()
    savefig("Figure_3_calibration_robustness")


# -----------------------------
# Figure 4: ablation and baseline comparison
# FIXED: clear full-QC vs reduced-QC labels
# -----------------------------
def make_figure4():
    abl = pd.read_csv(TABLES / "ablation_results.csv")
    base = pd.read_csv(TABLES / "baseline_comparison_results.csv")
    red = pd.read_csv(TABLES / "reduced_qc_baseline_results.csv")

    abl_plot = abl[["Model", "AUROC", "AUPRC", "f1"]].copy()
    base_plot = base[["Model", "AUROC", "AUPRC", "f1"]].copy()
    red_plot = red[["Model", "AUROC", "AUPRC", "f1"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: ablation
    ax = axes[0]
    x = np.arange(len(abl_plot))
    width = 0.25
    ax.bar(x - width, abl_plot["AUROC"], width, label="AUROC")
    ax.bar(x, abl_plot["AUPRC"], width, label="AUPRC")
    ax.bar(x + width, abl_plot["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(abl_plot["Model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Ablation analysis")
    ax.legend()

    # Panel B: full vs reduced QC baselines
    base_order = [
        "Logistic_regression_QC",
        "Random_forest_QC",
        "Gradient_boosting_QC",
    ]
    red_order = [
        "Logistic_regression_reduced_QC",
        "Random_forest_reduced_QC",
        "Gradient_boosting_reduced_QC",
    ]

    base_plot = base_plot.set_index("Model").loc[base_order].reset_index()
    red_plot = red_plot.set_index("Model").loc[red_order].reset_index()

    comp = pd.concat([base_plot, red_plot], ignore_index=True)
    labels = ["Full LR", "Full RF", "Full GB", "Reduced LR", "Reduced RF", "Reduced GB"]

    ax = axes[1]
    x = np.arange(len(comp))
    width = 0.36
    ax.bar(x - width/2, comp["AUPRC"], width, label="AUPRC")
    ax.bar(x + width/2, comp["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Full-QC and reduced-QC baselines")
    ax.legend()

    # Visual separator and group labels
    ax.axvline(2.5, color="gray", linestyle="--", linewidth=1)
    ax.text(1.0, 1.02, "Full QC", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(4.0, 1.02, "Reduced QC", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    savefig("Figure_4_ablation_baselines")


# -----------------------------
# Figure 5: GBM external application and sensitivity
# FIXED: percentages shown above bars
# -----------------------------
def make_figure5():
    gbm = pd.read_csv(TABLES / "gbm_predictions.csv")
    clip3_path = TABLES / "gbm_predictions_clip3.csv"

    clip3 = pd.read_csv(clip3_path) if clip3_path.exists() else None

    prob_col = "rep_error_probability"
    pred_col = "rep_error_predicted"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: histogram
    ax = axes[0]
    ax.hist(gbm[prob_col], bins=60, alpha=0.8)
    ax.axvline(0.75, linestyle="--", lw=1.5, label="Threshold = 0.75")
    ax.set_xlabel("Predicted artifact-prioritization score")
    ax.set_ylabel("Number of GBM SNVs")
    ax.set_title("GBM score distribution")
    ax.legend()

    # Panel B: counts + percentages
    main_total = len(gbm)
    main_art = int(gbm[pred_col].sum())
    main_ret = main_total - main_art

    labels = ["Main\nartifact candidates", "Main\nretained"]
    counts = [main_art, main_ret]
    totals = [main_total, main_total]

    if clip3 is not None and pred_col in clip3.columns:
        clip_total = len(clip3)
        clip_art = int(clip3[pred_col].sum())
        clip_ret = clip_total - clip_art
        labels += ["Clip3\nartifact candidates", "Clip3\nretained"]
        counts += [clip_art, clip_ret]
        totals += [clip_total, clip_total]

    ax = axes[1]
    x = np.arange(len(labels))
    bars = ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of variants")
    ax.set_title("GBM main vs clip3 sensitivity")

    ymax = max(counts) * 1.15
    ax.set_ylim(0, ymax)

    for i, (bar, count, total) in enumerate(zip(bars, counts, totals)):
        pct = (count / total) * 100 if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.01,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    savefig("Figure_5_gbm_application")


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()