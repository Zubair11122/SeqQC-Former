#!/usr/bin/env python3
"""
Fixed Comparison Script with Proper Label Handling
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from matplotlib_venn import venn2, venn3
import sys
import pickle
from pathlib import Path
from typing import Tuple, Dict

# ===================== Visualization Settings =====================
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'pdf.fonttype': 42,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


# ===================== Data Loading =====================
def create_variant_key(row):
    """Create consistent variant key from different column naming conventions"""
    chrom = str(row.get('chrom', row.get('Chromosome', row.get('CHROM', ''))))
    pos = str(row.get('pos', row.get('Position', row.get('POS', ''))))
    ref = str(row.get('ref', row.get('Reference_Allele', row.get('REF', ''))))
    alt = str(row.get('alt', row.get('Tumor_Seq_Allele2', row.get('ALT', ''))))
    return f"{chrom}:{pos}:{ref}:{alt}"


def load_data(data_root: Path, mutect_path: Path, strelka_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load and merge all prediction data with robust error handling"""
    try:
        # Load SeqQC-Former predictions
        print("\nLoading SeqQC-Former predictions...")
        seqqc = pd.read_csv(data_root / "full_preds.csv")
        print("Columns in full_preds.csv:", seqqc.columns.tolist())

        if 'prob' not in seqqc.columns:
            if 'score' in seqqc.columns:
                seqqc['prob'] = seqqc['score']
            else:
                raise ValueError("SeqQC-Former file needs either 'prob' or 'score' column")

        seqqc['key'] = seqqc.apply(create_variant_key, axis=1)
        seqqc['TOOL'] = 'SeqQC-Former'

        # Load truth labels
        print("\nLoading truth labels...")
        with open(data_root / "variants_labeled.pkl", 'rb') as f:
            truth = pickle.load(f)

        print("Columns in variants_labeled.pkl:", truth.columns.tolist())
        truth = truth.rename(columns={
            "Chromosome": "chrom", "Start_Position": "pos",
            "Reference_Allele": "ref", "Tumor_Seq_Allele2": "alt",
            "seqc2_positive": "label"
        })
        truth["key"] = truth.apply(create_variant_key, axis=1)

        # Print label distribution for debugging
        print("\nLabel distribution in truth data:")
        print(truth['label'].value_counts(dropna=False))

        # Load variant callers' predictions
        def load_caller_data(path, tool_name):
            print(f"\nLoading {tool_name} data from {path}...")
            df = pd.read_csv(path)
            print(f"Columns in {tool_name} file:", df.columns.tolist())

            if 'prob' not in df.columns:
                if 'score' in df.columns:
                    df['prob'] = df['score']
                elif 'QUAL' in df.columns:
                    df['prob'] = 1 / (1 + np.exp(-df['QUAL'] / 10))
                else:
                    df['prob'] = 0.5  # Default if no score available

            df['key'] = df.apply(create_variant_key, axis=1)
            return df.assign(TOOL=tool_name)

        mutect = load_caller_data(mutect_path, 'Mutect2')
        strelka = load_caller_data(strelka_path, 'Strelka2')

        # Merge all data - FIXED MERGE LOGIC
        print("\nMerging data...")
        dfs = []
        for df in [seqqc, mutect, strelka]:
            # Always merge with truth labels to preserve all classes
            merged = df.merge(truth[['key', 'label']], on='key', how='left')
            dfs.append(merged[['key', 'label', 'prob', 'TOOL']])

        all_preds = pd.concat(dfs).drop_duplicates(subset=['key', 'TOOL'])

        # Check for missing labels
        missing_labels = all_preds['label'].isna().sum()
        if missing_labels > 0:
            print(f"Warning: {missing_labels} variants are missing labels", file=sys.stderr)
            all_preds = all_preds.dropna(subset=['label'])

        # Verify we have both classes after merging
        print("\nLabel distribution after merging:")
        print(all_preds['label'].value_counts(dropna=False))

        return all_preds, calculate_all_metrics(all_preds)

    except Exception as e:
        print(f"\nERROR loading data: {str(e)}", file=sys.stderr)
        print("\nRequired file structure:")
        print("- data_root/")
        print("  ├── full_preds.csv (columns: variant info + prob/score)")
        print("  ├── variants_labeled.pkl (columns: variant info + seqc2_positive)")
        print("  └── baseline_out/")
        print("      ├── mutect2_IL_1_bwa.csv")
        print("      └── strelka2_IL_1_bwa.csv")
        sys.exit(1)


def calculate_all_metrics(all_preds: pd.DataFrame) -> Dict:
    """Calculate metrics for all tools with robust edge case handling"""
    metrics = {}
    for tool in ['SeqQC-Former', 'Mutect2', 'Strelka2']:
        df = all_preds[all_preds['TOOL'] == tool]
        if len(df) == 0:
            print(f"Warning: No data found for {tool}", file=sys.stderr)
            continue

        y_true = df['label'].values
        y_score = df['prob'].values

        # Verify we have both classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            print(f"\nWarning: Only {len(unique_classes)} class(es) present for {tool}")
            pos_rate = y_true.mean() if len(y_true) > 0 else 0
            metrics[tool] = {
                'AUROC': np.nan,
                'AUPRC': np.nan,
                'F1': 0,
                'Precision': 0,
                'Recall': 0,
                'Specificity': 1 if len(y_true) > 0 and unique_classes[0] == 0 else 0,
                'N': len(y_true),
                'Positive_Rate': pos_rate,
                'Warning': f'Only {len(unique_classes)} class in labels'
            }
            continue

        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Calculate PR metrics
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        # Calculate binary metrics with adjusted threshold
        y_pred = (y_score >= 0.5).astype(int)

        metrics[tool] = {
            'AUROC': roc_auc,
            'AUPRC': pr_auc,
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'N': len(y_true),
            'Positive_Rate': y_true.mean()
        }
    return metrics


# [Rest of the visualization functions remain the same as in previous version]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare SeqQC-Former with Mutect2 and Strelka2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_root', type=Path, default='/home/zubair/Project/data_root',
                        help='Directory containing input files')
    parser.add_argument('--mutect', type=Path,
                        default='/home/zubair/Project/data_root/baseline_out/mutect2_IL_1_bwa.csv',
                        help='Path to Mutect2 results')
    parser.add_argument('--strelka', type=Path,
                        default='/home/zubair/Project/data_root/baseline_out/strelka2_IL_1_bwa.csv',
                        help='Path to Strelka2 results')
    parser.add_argument('--outdir', type=Path,
                        default='/home/zubair/Project/data_root/baseline_out/comparison_results',
                        help='Output directory')

    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    print("\n[1/6] Loading data...")
    all_preds, metrics = load_data(args.data_root, args.mutect, args.strelka)

    print("\n[2/6] Generating visualizations:")
    plots = [
        ("ROC Curve", plot_roc),
        ("PR Curve", plot_pr),
        ("Score Distributions", plot_score_distributions),
        ("Confusion Matrices", plot_confusion_matrices),
        ("Venn Diagrams", plot_venn)
    ]

    for name, func in plots:
        print(f"- {name}")
        func(all_preds, args.outdir)

    print("\n[3/6] Saving performance metrics...")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(args.outdir / 'performance_metrics.csv', float_format='%.4f')

    print("\n[4/6] Generating LaTeX table...")
    latex_table = metrics_df[['AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Specificity']].to_latex(
        float_format="%.3f",
        bold_rows=True,
        caption="Performance comparison between SeqQC-Former and baseline methods",
        label="tab:performance"
    )
    with open(args.outdir / 'performance_table.tex', 'w') as f:
        f.write(latex_table)

    print("\n[5/6] Generating markdown report...")
    markdown_content = (
        "# SeqQC-Former Comparison Results\n\n"
        "## Performance Metrics\n"
        f"```\n{metrics_df.to_string()}\n```\n\n"
        "## Generated Figures\n"
        "- ROC_Comparison.pdf\n"
        "- PR_Curve.pdf\n"
        "- Score_Distributions.pdf\n"
        "- CM_*.pdf (Confusion matrices)\n"
        "- Venn_All.pdf\n\n"
        "## Analysis Parameters\n"
        f"- Data root: {args.data_root}\n"
        f"- Mutect2 input: {args.mutect}\n"
        f"- Strelka2 input: {args.strelka}\n"
        f"- Output directory: {args.outdir}\n"
    )

    with open(args.outdir / 'README.md', 'w') as f:
        f.write(markdown_content)

    print("\n[6/6] Results summary:")
    print(f"\n{'Tool':<15} {'AUROC':<8} {'AUPRC':<8} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Specificity':<10}")
    print("-" * 65)
    for tool, vals in metrics.items():
        print(
            f"{tool:<15} {vals.get('AUROC', np.nan):.3f}   {vals.get('AUPRC', np.nan):.3f}   {vals.get('F1', 0):.3f}   {vals.get('Precision', 0):.3f}      {vals.get('Recall', 0):.3f}      {vals.get('Specificity', 0):.3f}")

    print(f"\nResults saved to: {args.outdir}")
    print("Generated files:")
    for f in sorted(args.outdir.glob('*')):
        print(f"  - {f.name}")