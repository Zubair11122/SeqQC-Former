#!/usr/bin/env python3
"""
20_collect_reviewer_safe_results.py

Collect metrics from reviewer-safe run folders into manuscript-ready tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="reviewer_safe/runs")
    ap.add_argument("--out-dir", default="reviewer_safe/publication_tables")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
        with open(metrics_file) as f:
            m = json.load(f)
        for split in ["val", "test"]:
            if split in m:
                row = {"run_name": m.get("run_name"), "feature_mode": m.get("feature_mode"), "split": split}
                row.update(m[split])
                rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {runs_dir}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reviewer_safe_model_comparison.csv", index=False)

    keep = [c for c in ["run_name", "feature_mode", "split", "n", "n_positive", "auroc", "auprc", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "threshold"] if c in df.columns]
    print("✅ Collected reviewer-safe results")
    print(df[keep].to_string(index=False))
    print(f"Saved: {out_dir / 'reviewer_safe_model_comparison.csv'}")


if __name__ == "__main__":
    main()
