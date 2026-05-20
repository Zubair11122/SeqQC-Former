from pathlib import Path
import shutil
import pandas as pd
import numpy as np

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU")
ROOT = PROJECT_ROOT / "final_publication_results"

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
PUB_FIGURES = ROOT / "publication_figures"
GBM_FIGURES = PROJECT_ROOT / "data_roots" / "gbm" / "figures"

MAIN_TABLE_DIR = ROOT / "publication_main_tables"
SUPP_TABLE_DIR = ROOT / "publication_supplementary_tables"
SUPP_FIG_DIR = ROOT / "publication_supplementary_figures"

MAIN_TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUPP_TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_file(folder, candidates, extensions=None):
    """
    Find file by exact name first, then relaxed stem matching.
    """
    folder = Path(folder)

    if extensions is None:
        extensions = [""]

    for name in candidates:
        p = folder / name
        if p.exists():
            return p

    for name in candidates:
        stem = Path(name).stem
        stem_clean = (
            stem.replace("(1)", "")
            .replace("(2)", "")
            .replace("(3)", "")
            .replace(" ", "_")
        )

        for ext in extensions:
            matches = sorted(folder.glob(f"{stem_clean}*{ext}"))
            if matches:
                return matches[0]

    return None


def find_file_recursive(search_dirs, candidates, extensions):
    """
    Search exact names and relaxed names across multiple folders recursively.
    """
    for folder in search_dirs:
        folder = Path(folder)
        if not folder.exists():
            continue

        # Exact search in this folder
        for name in candidates:
            p = folder / name
            if p.exists():
                return p

        # Recursive exact search
        for name in candidates:
            matches = sorted(folder.rglob(name))
            if matches:
                return matches[0]

        # Recursive relaxed stem search
        for name in candidates:
            stem = Path(name).stem
            stem_clean = (
                stem.replace("(1)", "")
                .replace("(2)", "")
                .replace("(3)", "")
                .replace(" ", "_")
            )

            for ext in extensions:
                matches = sorted(folder.rglob(f"*{stem_clean}*{ext}"))
                if matches:
                    return matches[0]

    return None


def safe_read_csv(folder, candidates, required=True):
    p = find_file(folder, candidates, extensions=[".csv"])
    if p is None:
        msg = f"Missing CSV. Tried: {candidates}"
        if required:
            raise FileNotFoundError(msg)
        print("WARNING:", msg)
        return None, None

    print("Reading:", p)
    return pd.read_csv(p), p


def save_csv(df, path, item_name=None, description=None):
    df.to_csv(path, index=False)
    print("Saved:", path)

    if item_name:
        manifest_rows.append({
            "Item": item_name,
            "Output file": str(path),
            "Status": "saved",
            "Description": description if description else ""
        })


def round_numeric(df, digits=6):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def copy_supp_figure(candidates, out_name, item_name, description):
    """
    Copy supplementary figure from likely locations.
    Searches deeply across project folders.
    """
    search_dirs = [
        FIGURES,
        PUB_FIGURES,
        ROOT,
        GBM_FIGURES,
        PROJECT_ROOT,
    ]

    extensions = [".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"]

    src = find_file_recursive(search_dirs, candidates, extensions)

    if src is None:
        print(f"WARNING: Could not find {item_name}. Tried: {candidates}")
        manifest_rows.append({
            "Item": item_name,
            "Output file": str(SUPP_FIG_DIR / out_name),
            "Status": "missing",
            "Description": description
        })
        return None

    dst = SUPP_FIG_DIR / out_name
    shutil.copy2(src, dst)
    print(f"Copied {item_name}: {src} -> {dst}")

    manifest_rows.append({
        "Item": item_name,
        "Output file": str(dst),
        "Status": "saved",
        "Description": description
    })

    return dst


# =============================================================================
# MAIN TABLE 1
# Dataset and feature summary
# =============================================================================

table1 = pd.DataFrame([
    {"Category": "Dataset", "Item": "Total SEQC2-derived loci", "Value": "89,447"},
    {"Category": "Dataset", "Item": "Positive loci", "Value": "1,378"},
    {"Category": "Dataset", "Item": "Negative loci", "Value": "88,069"},
    {"Category": "Feature matrix", "Item": "Sequence window", "Value": "501 bp"},
    {"Category": "Feature matrix", "Item": "One-hot sequence shape", "Value": "4 × 501"},
    {"Category": "Feature matrix", "Item": "Structured QC features", "Value": "11"},
    {"Category": "Split", "Item": "Training loci", "Value": "57,245"},
    {"Category": "Split", "Item": "Validation loci", "Value": "14,312"},
    {"Category": "Split", "Item": "Held-out test loci", "Value": "17,890"},
    {"Category": "Held-out test", "Item": "Test positives", "Value": "276"},
    {"Category": "Held-out test", "Item": "Test negatives", "Value": "17,614"},
])

save_csv(
    table1,
    MAIN_TABLE_DIR / "Table_1_dataset_and_feature_summary.csv",
    item_name="Table 1",
    description="Dataset, feature matrix, and train/validation/test split summary"
)


# =============================================================================
# MAIN TABLE 2
# Held-out test performance and calibration
# =============================================================================

cal_metrics, _ = safe_read_csv(TABLES, ["calibration_metrics.csv"])


def get_cal_metric(name):
    val = cal_metrics.loc[cal_metrics["metric"] == name, "value"]
    return float(val.iloc[0]) if len(val) else np.nan


table2 = pd.DataFrame([
    {"Section": "Held-out test performance", "Metric": "AUROC", "Value": 0.999970},
    {"Section": "Held-out test performance", "Metric": "AUPRC", "Value": 0.998202},
    {"Section": "Held-out test performance", "Metric": "Accuracy", "Value": 0.998155},
    {"Section": "Held-out test performance", "Metric": "Precision", "Value": 0.893204},
    {"Section": "Held-out test performance", "Metric": "Recall", "Value": 1.000000},
    {"Section": "Held-out test performance", "Metric": "F1-score", "Value": 0.943590},
    {"Section": "Held-out test performance", "Metric": "True positives", "Value": 276},
    {"Section": "Held-out test performance", "Metric": "True negatives", "Value": 17581},
    {"Section": "Held-out test performance", "Metric": "False positives", "Value": 33},
    {"Section": "Held-out test performance", "Metric": "False negatives", "Value": 0},
    {"Section": "Held-out test performance", "Metric": "Decision threshold", "Value": 0.75},
    {"Section": "Calibration", "Metric": "Brier score", "Value": get_cal_metric("Brier score")},
    {"Section": "Calibration", "Metric": "Expected calibration error", "Value": get_cal_metric("Expected calibration error")},
    {"Section": "Calibration", "Metric": "Mean predicted probability", "Value": get_cal_metric("mean_predicted_probability")},
    {"Section": "Calibration", "Metric": "Median predicted probability", "Value": get_cal_metric("median_predicted_probability")},
    {"Section": "Calibration", "Metric": "Positive fraction", "Value": get_cal_metric("positive_fraction")},
])

table2 = round_numeric(table2)

save_csv(
    table2,
    MAIN_TABLE_DIR / "Table_2_test_performance_and_calibration.csv",
    item_name="Table 2",
    description="Held-out SEQC2 test performance and probability calibration summary"
)


# =============================================================================
# MAIN TABLE 3
# Robustness summary
# =============================================================================

robustness, _ = safe_read_csv(TABLES, [
    "robustness_test_only.csv",
    "robustness_test_only(1).csv",
    "robustness_test_only(2).csv",
    "robustness_test_only(3).csv",
])

noise_col = "noise_sigma" if "noise_sigma" in robustness.columns else robustness.columns[0]

keep_cols = [
    c for c in [
        noise_col,
        "AUROC",
        "AUPRC",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "TP",
        "TN",
        "FP",
        "FN"
    ]
    if c in robustness.columns
]

table3 = robustness[keep_cols].copy()
table3 = table3.rename(columns={
    noise_col: "QC noise sigma",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
})
table3 = round_numeric(table3)

save_csv(
    table3,
    MAIN_TABLE_DIR / "Table_3_robustness_under_QC_perturbation.csv",
    item_name="Table 3",
    description="Robustness of SeqQC-Former under simulated QC-feature perturbation"
)


# =============================================================================
# MAIN TABLE 4
# Ablation and baseline comparison
# =============================================================================

ablation, _ = safe_read_csv(TABLES, [
    "ablation_results.csv",
    "ablation_results(1).csv",
])

baseline, _ = safe_read_csv(TABLES, [
    "baseline_comparison_results.csv",
])

reduced, _ = safe_read_csv(TABLES, [
    "reduced_qc_baseline_results.csv",
])

ablation_main = ablation.copy()
ablation_main["Experiment"] = "Ablation"

baseline_main = baseline.copy()
baseline_main["Experiment"] = "Full-QC classical baseline"

reduced_main = reduced.copy()
reduced_main["Experiment"] = "Reduced-QC classical baseline"

table4 = pd.concat([ablation_main, baseline_main, reduced_main], ignore_index=True, sort=False)

preferred_cols = [
    "Experiment",
    "Model",
    "AUROC",
    "AUPRC",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "TP",
    "TN",
    "FP",
    "FN",
    "best_val_AUROC"
]

table4 = table4[[c for c in preferred_cols if c in table4.columns]].copy()
table4 = table4.rename(columns={
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "best_val_AUROC": "Best validation AUROC"
})
table4 = round_numeric(table4)

save_csv(
    table4,
    MAIN_TABLE_DIR / "Table_4_ablation_and_baseline_comparison.csv",
    item_name="Table 4",
    description="Ablation analysis and full/reduced-QC baseline comparison"
)


# =============================================================================
# SUPPLEMENTARY TABLE S1
# QC feature definitions
# =============================================================================

table_s1 = pd.DataFrame([
    {"Feature": "DP", "Definition": "Read depth at the candidate locus.", "Feature group": "Coverage"},
    {"Feature": "AD", "Definition": "Alternate-allele read depth at the candidate locus.", "Feature group": "Allele support"},
    {"Feature": "VAF", "Definition": "Variant allele fraction computed from alternate-allele support and depth.", "Feature group": "Allele support"},
    {"Feature": "MQ", "Definition": "Mapping-quality summary for reads supporting the locus.", "Feature group": "Read quality"},
    {"Feature": "SB", "Definition": "Strand-bias summary at the candidate locus.", "Feature group": "Bias"},
    {"Feature": "tumor_strand_bias", "Definition": "Tumor-specific strand-bias estimate.", "Feature group": "Bias"},
    {"Feature": "tumor_orientation_bias", "Definition": "Tumor-specific read-orientation bias estimate.", "Feature group": "Bias"},
    {"Feature": "tumor_clipped_fraction", "Definition": "Fraction of tumor reads with clipping at or near the candidate locus.", "Feature group": "Read quality"},
    {"Feature": "tumor_mismatch_fraction", "Definition": "Fraction of tumor read alignments containing mismatches near the candidate locus.", "Feature group": "Read quality"},
    {"Feature": "normal_alt_fraction", "Definition": "Alternate-allele fraction observed in the matched normal sample.", "Feature group": "Normal evidence"},
    {"Feature": "germline_support_flag", "Definition": "Indicator of normal/germline-like support at the candidate locus.", "Feature group": "Normal evidence"},
])

save_csv(
    table_s1,
    SUPP_TABLE_DIR / "Table_S1_QC_feature_definitions.csv",
    item_name="Table S1",
    description="Names and definitions of the 11 structured QC features"
)


# =============================================================================
# SUPPLEMENTARY TABLES S2-S10
# =============================================================================

supp_table_specs = [
    {
        "item": "Table S2",
        "out": "Table_S2_full_calibration_bins.csv",
        "candidates": ["calibration_bins.csv"],
        "description": "Full calibration-bin statistics"
    },
    {
        "item": "Table S3",
        "out": "Table_S3_full_robustness_results.csv",
        "candidates": ["robustness_test_only.csv", "robustness_test_only(3).csv"],
        "description": "Complete robustness results"
    },
    {
        "item": "Table S4",
        "out": "Table_S4_full_ablation_results.csv",
        "candidates": ["ablation_results.csv", "ablation_results(1).csv"],
        "description": "Full ablation metrics"
    },
    {
        "item": "Table S6",
        "out": "Table_S6_GBM_main_and_clip3_summary.csv",
        "candidates": ["final_manuscript_numbers.csv"],
        "description": "GBM main and clip3 summary rows from final manuscript numbers"
    },
    {
        "item": "Table S7",
        "out": "Table_S7_GBM_publication_summary.csv",
        "candidates": ["gbm_summary_for_publication.csv"],
        "description": "Overall and sample-level GBM publication summary"
    },
    {
        "item": "Table S8",
        "out": "Table_S8_GBM_gene_level_summary.csv",
        "candidates": ["gbm_gene_level_artifact_summary.csv"],
        "description": "Full gene-level GBM artifact-prioritization summary"
    },
    {
        "item": "Table S9",
        "out": "Table_S9_GBM_per_variant_artifact_prioritization_scores.csv",
        "candidates": ["gbm_predictions.csv"],
        "description": "Per-variant GBM artifact-prioritization scores"
    },
    {
        "item": "Table S10",
        "out": "Table_S10_GBM_clip3_per_variant_artifact_prioritization_scores.csv",
        "candidates": ["gbm_predictions_clip3.csv"],
        "description": "GBM clip3 sensitivity per-variant artifact-prioritization scores"
    },
]

for spec in supp_table_specs:
    df, src = safe_read_csv(TABLES, spec["candidates"], required=False)
    if df is None:
        placeholder = pd.DataFrame([{
            "Status": "missing",
            "Message": f"Could not find source file. Tried: {spec['candidates']}"
        }])
        save_csv(
            placeholder,
            SUPP_TABLE_DIR / spec["out"].replace(".csv", "_MISSING.csv"),
            item_name=spec["item"],
            description=spec["description"]
        )
    else:
        save_csv(
            round_numeric(df),
            SUPP_TABLE_DIR / spec["out"],
            item_name=spec["item"],
            description=spec["description"]
        )


# =============================================================================
# SUPPLEMENTARY TABLE S5
# Full baseline + reduced-QC baseline
# =============================================================================

baseline_s5 = baseline.copy()
baseline_s5["Feature_set"] = "Full_QC"

reduced_s5 = reduced.copy()
reduced_s5["Feature_set"] = "Reduced_QC"

table_s5 = pd.concat([baseline_s5, reduced_s5], ignore_index=True, sort=False)
table_s5 = round_numeric(table_s5)

save_csv(
    table_s5,
    SUPP_TABLE_DIR / "Table_S5_full_baseline_and_reduced_QC_results.csv",
    item_name="Table S5",
    description="Full baseline and reduced-QC baseline metrics"
)


# =============================================================================
# SUPPLEMENTARY FIGURES S1-S5
# =============================================================================

copy_supp_figure(
    candidates=[
        "calibration_curve.png",
        "calibration_curve.pdf",
        "Figure_3_calibration_robustness.png",
        "Figure_3_calibration_robustness.pdf"
    ],
    out_name="Figure_S1_full_calibration_reliability_curve.png",
    item_name="Figure S1",
    description="Full calibration reliability curve"
)

copy_supp_figure(
    candidates=[
        "probability_by_label_histogram.png",
        "probability_by_label_histogram.pdf",
        "Figure_2_test_performance.png",
        "Figure_2_test_performance.pdf"
    ],
    out_name="Figure_S2_held_out_score_distribution_by_label.png",
    item_name="Figure S2",
    description="Held-out score distribution by true label"
)

copy_supp_figure(
    candidates=[
        "top_artifact_genes.png",
        "top_artifact_genes.pdf",
        "gbm_top_artifact_genes.png",
        "GBM_top_artifact_genes.png"
    ],
    out_name="Figure_S3_top_GBM_genes_ranked_by_artifact_prioritization_burden.png",
    item_name="Figure S3",
    description="Top GBM genes ranked by artifact-prioritization burden"
)

copy_supp_figure(
    candidates=[
        "shap_summary_plot.png",
        "shap_summary_plot.pdf",
        "SHAP_summary_plot.png",
        "shap_summary.png",
        "qc_shap_summary.png"
    ],
    out_name="Figure_S4_QC_feature_contribution_SHAP_summary.png",
    item_name="Figure S4",
    description="QC feature contribution analysis / SHAP summary"
)

copy_supp_figure(
    candidates=[
        "sample_level_artifact_rate.png",
        "sample_level_artifact_rate.pdf",
        "gbm_sample_level_artifact_rate.png",
        "sample_artifact_summary.png",
        "gbm_sample_summary.png",
        "sample_level_gbm_summary.png"
    ],
    out_name="Figure_S5_sample_level_GBM_artifact_prioritization_summary.png",
    item_name="Figure S5",
    description="Sample-level GBM artifact-prioritization summary"
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = pd.DataFrame(manifest_rows)
save_csv(
    manifest,
    ROOT / "publication_table_figure_manifest.csv"
)

print("\nDONE.")
print("Main tables saved to:", MAIN_TABLE_DIR)
print("Supplementary tables saved to:", SUPP_TABLE_DIR)
print("Supplementary figures saved to:", SUPP_FIG_DIR)
print("Manifest saved to:", ROOT / "publication_table_figure_manifest.csv")