#!/usr/bin/env python3
"""
Compare SeqQC-Former with Mutect2 and Strelka2 - Publication Ready Figures
"""
from pathlib import Path
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
from typing import Tuple, Dict

# ===================== Visualization Settings =====================
plt.style.use('seaborn')
sns.set(font_scale=1.3)
plt.rcParams.update({
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12
})


# ===================== Data Loading =====================
def load_data(data_root: Path, mutect_path: Path, strelka_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load and merge all prediction data."""
    # Load SeqQC-Former predictions
    seqqc = pd.read_csv(data_root / "full_preds.csv")
    seqqc['TOOL'] = 'SeqQC-Former'

    # Load truth labels
    truth = pd.read_pickle(data_root / "variants_labeled.pkl")
    truth = truth.rename(columns={
        "Chromosome": "chrom", "Start_Position": "pos",
        "Reference_Allele": "ref", "Tumor_Seq_Allele2": "alt",
        "seqc2_positive": "label"
    })
    truth["key"] = truth["chrom"].astype(str) + ":" + truth["pos"].astype(str) + ":" + truth["ref"] + ":" + truth["alt"]

    # Load variant callers' predictions
    mutect = pd.read_csv(mutect_path).assign(TOOL='Mutect2')
    strelka = pd.read_csv(strelka_path).assign(TOOL='Strelka2')

    # Merge all tools
    all_preds = pd.concat([
        seqqc[['key', 'label', 'prob', 'TOOL']],
        mutect,
        strelka
    ])

    # Calculate metrics
    metrics = {}
    for tool in all_preds['TOOL'].unique():
        df = all_preds[all_preds['TOOL'] == tool]
        metrics[tool] = calculate_metrics(df['label'], df['prob'])

    return all_preds, metrics


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Calculate comprehensive performance metrics."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        'AUROC': auc(fpr, tpr),
        'AUPRC': auc(recall, precision),
        'F1': f1_score(y_true, (y_score >= 0.5).astype(int)),
        'Precision': precision_score(y_true, (y_score >= 0.5).astype(int)),
        'Recall': recall_score(y_true, (y_score >= 0.5).astype(int)),
        'Specificity': recall_score(y_true, (y_score >= 0.5).astype(int), pos_label=0)
    }


# ===================== Visualization Functions =====================
def plot_roc(all_preds: pd.DataFrame, outdir: Path):
    """Generate ROC curve comparison."""
    plt.figure(figsize=(8, 6))
    for tool in ['SeqQC-Former', 'Mutect2', 'Strelka2']:
        df = all_preds[all_preds['TOOL'] == tool]
        fpr, tpr, _ = roc_curve(df['label'], df['prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3,
                 label=f'{tool} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(outdir / 'ROC_Comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_pr(all_preds: pd.DataFrame, outdir: Path):
    """Generate Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    for tool in ['SeqQC-Former', 'Mutect2', 'Strelka2']:
        df = all_preds[all_preds['TOOL'] == tool]
        precision, recall, _ = precision_recall_curve(df['label'], df['prob'])
        plt.plot(recall, precision, lw=3,
                 label=f'{tool} (AP = {auc(recall, precision):.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(outdir / 'PR_Curve.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_score_distributions(all_preds: pd.DataFrame, outdir: Path):
    """Generate violin plots of prediction distributions."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=all_preds,
        x='TOOL',
        y='prob',
        hue='label',
        split=True,
        palette={0: '#1f77b4', 1: '#ff7f0e'},
        inner='quartile',
        cut=0,
        order=['SeqQC-Former', 'Mutect2', 'Strelka2']
    )
    plt.ylabel('Prediction Score')
    plt.xlabel('')
    plt.title('Score Distributions by True Class')
    plt.legend(title='True Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'Score_Distributions.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrices(all_preds: pd.DataFrame, outdir: Path):
    """Generate normalized confusion matrices."""
    for tool in ['SeqQC-Former', 'Mutect2', 'Strelka2']:
        df = all_preds[all_preds['TOOL'] == tool]
        pred = (df['prob'] >= 0.5).astype(int)
        cm = confusion_matrix(df['label'], pred, normalize='true')

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'],
            cbar=False,
            annot_kws={'size': 12}
        )
        plt.title(f'{tool} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(outdir / f'CM_{tool}.pdf', bbox_inches='tight', dpi=300)
        plt.close()


def plot_venn(all_preds: pd.DataFrame, outdir: Path):
    """Generate Venn diagrams of variant calls."""
    tools = {
        'SeqQC-Former': set(all_preds[(all_preds['TOOL'] == 'SeqQC-Former') & (all_preds['prob'] >= 0.5)]['key']),
        'Mutect2': set(all_preds[(all_preds['TOOL'] == 'Mutect2') & (all_preds['prob'] >= 0.5)]['key']),
        'Strelka2': set(all_preds[(all_preds['TOOL'] == 'Strelka2') & (all_preds['prob'] >= 0.5)]['key'])
    }

    # 3-way Venn
    plt.figure(figsize=(8, 6))
    venn3(
        [tools['SeqQC-Former'], tools['Mutect2'], tools['Strelka2']],
        set_labels=['SeqQC-Former', 'Mutect2', 'Strelka2'],
        set_colors=['#2ca02c', '#1f77b4', '#d62728'],
        alpha=0.7
    )
    plt.title('Variant Call Overlap (Threshold=0.5)')
    plt.savefig(outdir / 'Venn_All.pdf', bbox_inches='tight', dpi=300)
    plt.close()


# ===================== Main Execution =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare SeqQC-Former with Mutect2 and Strelka2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_root', type=Path, required=True,
                        help='Directory containing full_preds.csv and variants_labeled.pkl')
    parser.add_argument('--mutect', type=Path, required=True,
                        help='Path to Mutect2 predictions CSV')
    parser.add_argument('--strelka', type=Path, required=True,
                        help='Path to Strelka2 predictions CSV')
    parser.add_argument('--outdir', type=Path, default='comparison_results',
                        help='Output directory for figures')

    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    print("Loading data...")
    all_preds, metrics = load_data(args.data_root, args.mutect, args.strelka)

    print("Generating visualizations:")
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

    # Save metrics
    pd.DataFrame.from_dict(metrics, orient='index').to_csv(args.outdir / 'performance_metrics.csv')

    print(f"\nResults saved to: {args.outdir}")
    print("Generated files:")
    for f in sorted(args.outdir.glob('*')):
        print(f"  - {f.name}")