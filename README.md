# SeqQC-Former: Fusing Sequence Context and Read-Quality for Calibrated Somatic SNV Detection

This repository contains the full implementation of **SeqQC-Former**, an interpretable deep-learning framework that integrates *sequence context* and *read-quality metrics* for calibrated somatic SNV detection.

---

## 🧩 Overview

SeqQC-Former fuses:
- Local nucleotide sequence context,
- Read-quality metrics extracted from tumor BAMs,
- Replication-timing and mappability features,
- and Transformer-based feature learning,

to yield high-confidence, calibrated SNV predictions that outperform traditional variant callers such as **Mutect2** and **Strelka2**.

---

## 📂 Repository Structure

SeqQC-Former/
│
├── README.md
├── config.yaml
├── common.env
├── environment.yml
├── .gitignore
│
├── src/
│ ├── 01_merge_and_label.py
│ ├── 02_qc_make_sites_and_windows.py
│ ├── 03_qc_extract_qc_from_bam.py
│ ├── 04_qc_merge_qc_bigwig.py
│ ├── 05_make_features_qc.py
│ ├── 06_train_model.py
│ ├── 07_eval_full.py
│ ├── 08_eval_balanced.py
│ ├── 09_find_best_threshold.py
│ ├── 10_analyze_thresholds.py
│ ├── 11_export_balanced_preds.py
│ ├── 12_make_keys.py
│ ├── 13_clean_preds.py
│ ├── 14_vcf_to_csv.py
│ ├── 15_compare_tools.py
│ ├── 16_bootstrap_ci.py
│ └── utils/
│
├── dataset/
│ ├── variants_labeled.pkl
│ ├── sites_win.bed
│ ├── umap.tab
│ ├── rtim.tab
│ ├── qc_bam.csv
│ └── qc_merged.csv
│
└── output/
├── features.h5
├── rep_error_net.ckpt
├── eval_full_metrics.txt
├── full_preds.csv
├── balanced_preds.csv
├── best_threshold.txt
├── threshold_sweep.csv
├── full_preds_with_keys.csv
├── full_preds_clean_by_key.csv
└── comparison_results/
├── ROC_Comparison.pdf
├── PR_Curve.pdf
├── Score_Distributions.pdf
├── CM_SeqQC-Former.pdf
├── Venn_All.pdf
├── performance_metrics.csv
├── performance_table.tex
└── README.md

---

## ⚙️ Environment Setup

All necessary paths and references are declared in:
- **`config.yaml`** → defines `data_root`, reference genome, and file paths.
- **`common.env`** → defines environment variables for `REF`, `HC`, `TUM`, `NORM`, and `OUT`.

### Example (`common.env`)
```bash
REF="./data_root/reference/GRCh38.fa"
HC="./data_root/hc_regions.bed"
TUM="./data_root/bam_tumor/T1.bam"
NORM="./data_root/bam_normal/N1.bam"
OUT="./data_root/baseline_out"
conda env create -f environment.yml
conda activate seqqc-former

Create Conda Environment

conda env create -f environment.yml
conda activate seqqc-former

Pipeline Execution

Run the scripts in order from 01 → 16:

Step	Script	Description	Key Output
1️	01_merge_and_label.py	Merge tumor/normal MAFs + SEQC2 VCFs and assign truth labels	variants_labeled.pkl
2️	02_qc_make_sites_and_windows.py	Generate variant sites & genomic windows	sites.csv, sites_win.bed
3️	03_qc_extract_qc_from_bam.py	Extract BAM-based QC metrics (DP, MQ, VAF, SB)	qc_bam.csv
4️	04_qc_merge_qc_bigwig.py	Merge QC tables with UMAP & replication timing	qc_merged.csv
5️	05_make_features_qc.py	Build HDF5 feature set (sequence + QC)	features.h5
6️	06_train_model.py	Train Transformer model (PyTorch Lightning)	rep_error_net.ckpt
7️	07_eval_full.py	Evaluate full dataset	full_preds.csv
8️	08_eval_balanced.py	Evaluate balanced subset	metrics printed
9️	09_find_best_threshold.py	Find best F1 threshold	best_threshold.txt
🔟	10_analyze_thresholds.py	Sweep thresholds, plot confusion matrices	threshold_sweep.csv
11️	11_export_balanced_preds.py	Export balanced predictions	balanced_preds.csv
12️	12_make_keys.py	Map variant keys (chrom:pos:ref:alt)	full_preds_with_keys.csv
13️	13_clean_preds.py	Merge SeqQC with truth and clean duplicates	full_preds_clean_by_key.csv
14️	14_vcf_to_csv.py	Convert Mutect2/Strelka2 VCFs → CSV	baseline_out/*.csv
15️	15_compare_tools.py	Compare SeqQC-Former vs. Mutect2 vs. Strelka2	ROC/PR plots + metrics
16️	16_bootstrap_ci.py	Bootstrap confidence intervals for AUC	CI tables

All final figures and tables appear in:

output/comparison_results/

Example Quickstart
# 1. Prepare data
python src/01_merge_and_label.py
python src/02_qc_make_sites_and_windows.py

# 2. Extract and merge quality features
python src/03_qc_extract_qc_from_bam.py
python src/04_qc_merge_qc_bigwig.py
python src/05_make_features_qc.py

# 3. Train and evaluate model
python src/06_train_model.py
python src/07_eval_full.py
python src/09_find_best_threshold.py

# 4. Compare against external callers
python src/14_vcf_to_csv.py
python src/15_compare_tools.py

Outputs

rep_error_net.ckpt — trained checkpoint

full_preds.csv / balanced_preds.csv — predictions

comparison_results/ — ROC, PR, confusion matrices, Venn diagrams, and tables

performance_table.tex — LaTeX summary for publication

Citation

Muhammad Zubair et al.
SeqQC-Former: Fusing Sequence Context and Read-Quality for Calibrated Somatic SNV Detection, 2025-2026.

License

This project is distributed under the MIT License.
See LICENSE for details.
