#!/usr/bin/env python3
"""
13_summarize_gbm_publication_tables_updated.py

Create publication-style summaries and cleaned/removed MAF-like tables from
GBM prediction output.

Keeps your original output names but adds stronger validation and threshold
metadata.
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np


CFG = Path("config.yaml")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(CFG)
    root = Path(cfg["data_root"])
    out_dir = root / "gbm"

    pred_file = out_dir / "gbm_predictions.csv"
    out_summary = out_dir / "gbm_summary_for_publication.csv"
    out_gene = out_dir / "gbm_gene_level_artifact_summary.csv"
    out_removed = out_dir / "gbm_removed_likely_artifacts.maf"
    out_clean = out_dir / "gbm_clean_somatic_maf.maf"

    threshold = float(cfg.get("gbm_prediction_threshold", 0.75))

    if not pred_file.exists():
        raise FileNotFoundError(f"Missing {pred_file}. Run 12_predict_gbm_rep_errors_updated.py first.")

    df = pd.read_csv(pred_file)
    required = ["key", "rep_error_probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    df["rep_error_predicted"] = (df["rep_error_probability"] >= threshold).astype(int)

    summary_rows = [
        {
            "level": "overall",
            "group": "all",
            "threshold": threshold,
            "n_variants": len(df),
            "n_predicted_artifact": int(df["rep_error_predicted"].sum()),
            "artifact_fraction": float(df["rep_error_predicted"].mean()) if len(df) else 0.0,
            "median_rep_error_probability": float(df["rep_error_probability"].median()) if len(df) else 0.0,
            "mean_rep_error_probability": float(df["rep_error_probability"].mean()) if len(df) else 0.0,
        }
    ]

    groupings = {
        "sample": "Tumor_Sample_Barcode",
        "source_maf": "source_maf",
        "variant_classification": "Variant_Classification",
    }

    for level, col in groupings.items():
        if col not in df.columns:
            continue
        tmp = (
            df.groupby(col, dropna=False)
            .agg(
                n_variants=("key", "count"),
                n_predicted_artifact=("rep_error_predicted", "sum"),
                artifact_fraction=("rep_error_predicted", "mean"),
                median_rep_error_probability=("rep_error_probability", "median"),
                mean_rep_error_probability=("rep_error_probability", "mean"),
            )
            .reset_index()
        )
        for _, r in tmp.iterrows():
            summary_rows.append(
                {
                    "level": level,
                    "group": r[col],
                    "threshold": threshold,
                    "n_variants": int(r["n_variants"]),
                    "n_predicted_artifact": int(r["n_predicted_artifact"]),
                    "artifact_fraction": float(r["artifact_fraction"]),
                    "median_rep_error_probability": float(r["median_rep_error_probability"]),
                    "mean_rep_error_probability": float(r["mean_rep_error_probability"]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_summary, index=False)

    if "Hugo_Symbol" in df.columns:
        gene = (
            df.groupby("Hugo_Symbol", dropna=False)
            .agg(
                n_variants=("key", "count"),
                n_predicted_artifact=("rep_error_predicted", "sum"),
                artifact_fraction=("rep_error_predicted", "mean"),
                median_rep_error_probability=("rep_error_probability", "median"),
                mean_rep_error_probability=("rep_error_probability", "mean"),
            )
            .reset_index()
            .sort_values(["n_predicted_artifact", "artifact_fraction"], ascending=False)
        )
        gene.insert(1, "threshold", threshold)
        gene.to_csv(out_gene, index=False)
        print("Saved:", out_gene)

    removed = df[df["rep_error_predicted"] == 1].copy()
    clean = df[df["rep_error_predicted"] == 0].copy()

    removed.to_csv(out_removed, sep="\t", index=False)
    clean.to_csv(out_clean, sep="\t", index=False)

    print("Saved:", out_summary)
    print("Saved:", out_removed)
    print("Saved:", out_clean)
    print("\nOverall:")
    print(summary[summary["level"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
