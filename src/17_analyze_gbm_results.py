#!/usr/bin/env python3
"""
14_analyze_gbm_results_updated.py

Analysis of GBM prediction results.

Fixes/improvements:
  1. Uses gbm_prediction_threshold from config instead of hard-coded 0.75.
  2. Does not claim a real external artifact database unless one is provided.
  3. Saves sample/gene/probability summaries and figures.
  4. Adds optional artifact_gene_file config support.

Optional config entry:
  artifact_gene_file: /path/to/artifact_genes.txt

The file can be one gene per line or a CSV/TSV with a Hugo_Symbol/gene column.
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CFG = Path("config.yaml")
REPLICATION_ERROR_MOTIFS = ["GAA", "GAG", "CAG", "CTG", "CGG", "CCG", "TGG", "TGC"]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_artifact_genes(path) -> set[str]:
    if not path:
        return set()
    path = Path(path)
    if not path.exists():
        print(f"WARNING: artifact_gene_file not found: {path}")
        return set()

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        for col in ["Hugo_Symbol", "gene", "Gene", "symbol", "Symbol"]:
            if col in df.columns:
                return set(df[col].dropna().astype(str))
        raise ValueError(f"No gene column found in {path}")

    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def save_histogram(series, path: Path, threshold: float):
    plt.figure(figsize=(9, 6))
    plt.hist(series, bins=50, alpha=0.8, edgecolor="black")
    plt.axvline(x=threshold, linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Predicted replication-error probability")
    plt.ylabel("Number of variants")
    plt.title("GBM prediction probability distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_cdf(series, path: Path, threshold: float):
    plt.figure(figsize=(9, 6))
    sorted_probs = np.sort(series.to_numpy())
    cumulative = np.arange(1, len(sorted_probs) + 1) / max(len(sorted_probs), 1)
    plt.plot(sorted_probs, cumulative)
    plt.axvline(x=threshold, linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Predicted replication-error probability")
    plt.ylabel("Cumulative fraction")
    plt.title("GBM prediction cumulative distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    cfg = load_config(CFG)
    root = Path(cfg["data_root"])
    out_dir = root / "gbm"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "gbm_predictions.csv"
    threshold = float(cfg.get("gbm_prediction_threshold", 0.75))

    if not results_file.exists():
        raise FileNotFoundError(f"Missing {results_file}. Run prediction first.")

    df = pd.read_csv(results_file)
    if "rep_error_probability" not in df.columns:
        raise ValueError("gbm_predictions.csv must contain rep_error_probability")
    if "key" not in df.columns:
        raise ValueError("gbm_predictions.csv must contain key")

    df["rep_error_predicted"] = (df["rep_error_probability"] >= threshold).astype(int)

    print("=" * 70)
    print("GBM results analysis")
    print("=" * 70)
    print(f"Variants: {len(df):,}")
    print(f"Threshold: {threshold}")
    print(f"Predicted artifacts: {int(df['rep_error_predicted'].sum()):,} ({df['rep_error_predicted'].mean():.2%})")

    # Overall probability summary
    prob_summary = df["rep_error_probability"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    prob_summary.to_csv(out_dir / "gbm_probability_summary.csv")
    print(f"Saved: {out_dir / 'gbm_probability_summary.csv'}")

    save_histogram(df["rep_error_probability"], fig_dir / "probability_histogram.png", threshold)
    save_cdf(df["rep_error_probability"], fig_dir / "probability_cdf.png", threshold)
    print(f"Saved figures in: {fig_dir}")

    # Per-sample summary
    if "Tumor_Sample_Barcode" in df.columns:
        sample_summary = (
            df.groupby("Tumor_Sample_Barcode", dropna=False)
            .agg(
                n_variants=("key", "count"),
                n_artifacts=("rep_error_predicted", "sum"),
                artifact_rate=("rep_error_predicted", "mean"),
                mean_probability=("rep_error_probability", "mean"),
                median_probability=("rep_error_probability", "median"),
            )
            .sort_values(["n_artifacts", "artifact_rate"], ascending=False)
        )
        sample_summary.to_csv(out_dir / "detailed_sample_artifacts.csv")
        print(f"Saved: {out_dir / 'detailed_sample_artifacts.csv'}")

        plt.figure(figsize=(9, 6))
        plt.hist(sample_summary["artifact_rate"], bins=30, alpha=0.8, edgecolor="black")
        plt.xlabel("Predicted artifact rate per sample")
        plt.ylabel("Number of samples")
        plt.title("Distribution of predicted artifact rates across GBM samples")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "sample_artifact_rate_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Gene summary
    if "Hugo_Symbol" in df.columns:
        gene_summary = (
            df.groupby("Hugo_Symbol", dropna=False)
            .agg(
                n_variants=("key", "count"),
                n_artifacts=("rep_error_predicted", "sum"),
                artifact_rate=("rep_error_predicted", "mean"),
                mean_probability=("rep_error_probability", "mean"),
                median_probability=("rep_error_probability", "median"),
            )
            .reset_index()
            .sort_values(["n_artifacts", "artifact_rate"], ascending=False)
        )
        gene_summary.to_csv(out_dir / "detailed_gene_artifacts.csv", index=False)
        print(f"Saved: {out_dir / 'detailed_gene_artifacts.csv'}")

        top = gene_summary[gene_summary["n_variants"] >= 5].head(20)
        if len(top):
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top)), top["n_artifacts"].to_numpy())
            plt.yticks(range(len(top)), top["Hugo_Symbol"].astype(str).to_list())
            plt.xlabel("Number of predicted artifacts")
            plt.title("Top genes by predicted replication-error artifacts")
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(fig_dir / "top_artifact_genes.png", dpi=150, bbox_inches="tight")
            plt.close()

        artifact_genes = load_artifact_genes(cfg.get("artifact_gene_file"))
        if artifact_genes:
            df["in_external_artifact_gene_list"] = df["Hugo_Symbol"].astype(str).isin(artifact_genes)
            comparison = (
                df.groupby("in_external_artifact_gene_list")
                .agg(
                    n_variants=("key", "count"),
                    n_artifacts=("rep_error_predicted", "sum"),
                    artifact_rate=("rep_error_predicted", "mean"),
                    mean_probability=("rep_error_probability", "mean"),
                )
                .reset_index()
            )
            comparison.to_csv(out_dir / "external_artifact_gene_comparison.csv", index=False)
            print(f"Saved: {out_dir / 'external_artifact_gene_comparison.csv'}")
        else:
            print("No artifact_gene_file provided; skipping external artifact-gene comparison.")

    # Mutation motif based on ref/alt key only; this is limited and not true sequence-context motif analysis.
    def key_ref_alt_class(k: str) -> str:
        parts = str(k).split(":")
        if len(parts) >= 4:
            return f"{parts[2]}>{parts[3]}"
        return "unknown"

    df["substitution"] = df["key"].apply(key_ref_alt_class)
    subst_summary = (
        df.groupby("substitution")
        .agg(
            n_variants=("key", "count"),
            n_artifacts=("rep_error_predicted", "sum"),
            artifact_rate=("rep_error_predicted", "mean"),
            mean_probability=("rep_error_probability", "mean"),
        )
        .reset_index()
        .sort_values(["artifact_rate", "n_variants"], ascending=False)
    )
    subst_summary.to_csv(out_dir / "substitution_artifact_summary.csv", index=False)
    print(f"Saved: {out_dir / 'substitution_artifact_summary.csv'}")

    report_path = out_dir / "gbm_validation_report.txt"
    lines = [
        "=" * 70,
        "GBM External Validation Summary Report",
        "=" * 70,
        f"Total variants/site rows: {len(df):,}",
        f"Prediction threshold: {threshold}",
        f"Predicted artifacts: {int(df['rep_error_predicted'].sum()):,} ({df['rep_error_predicted'].mean():.2%})",
        f"Mean probability: {df['rep_error_probability'].mean():.4f}",
        f"Median probability: {df['rep_error_probability'].median():.4f}",
    ]

    if "Tumor_Sample_Barcode" in df.columns:
        lines.append(f"Number of samples: {df['Tumor_Sample_Barcode'].nunique():,}")
    if "Hugo_Symbol" in df.columns:
        top_gene_counts = df.groupby("Hugo_Symbol")["rep_error_predicted"].sum().sort_values(ascending=False).head(10)
        lines.append("Top genes by predicted artifact count:")
        for gene, count in top_gene_counts.items():
            lines.append(f"  {gene}: {int(count)}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")
    print("\n✅ Analysis complete")


if __name__ == "__main__":
    main()
