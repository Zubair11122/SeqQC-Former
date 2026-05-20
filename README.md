# SeqQC-Former

SeqQC-Former is a GPU-accelerated deep-learning pipeline for detecting, modeling, and analyzing replication-error-like somatic variant artifacts using sequence context, paired tumor-normal read-quality features, genomic annotations, and neural network-based classification.

This repository contains the updated public version of the SeqQC-Former project. The previous project information has been removed and replaced with the current replication-error analysis workflow.

## Overview

Replication-error-like artifacts and low-confidence somatic variant calls can affect downstream cancer genomics analyses. SeqQC-Former is designed to integrate multiple sources of evidence around candidate somatic SNV sites, including:

- local nucleotide sequence context
- paired tumor-normal BAM-derived read-quality features
- mapping quality and base quality features
- strand and allele-support information
- mappability annotations
- replication timing annotations
- deep-learning-based classification
- threshold optimization
- robustness testing
- SHAP-based model interpretation
- GBM cohort prediction and summary analysis

The project is intended for research use in computational genomics, cancer genomics, and somatic variant quality-control analysis.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
└── rep_error_project/
    ├── config.yaml
    ├── src/
    ├── data_roots/
    ├── final_publication_results/
    ├── checkpoints/
    ├── lightning_logs/
    ├── best_threshold.txt
    ├── roc_curves.png
    ├── validation_roc_curve.png
    ├── shap_summary_plot.png
    └── run_baseline_comparison.sh
Main Features
GPU-accelerated model training and evaluation
Sequence-window feature extraction around candidate SNV sites
Real paired tumor-normal BAM quality feature extraction
Mappability and replication-timing annotation integration
Chromosome-based train/validation/test splitting
Deep-learning model training with PyTorch Lightning
Test-set evaluation using optimized decision thresholds
Robustness analysis under reduced or perturbed QC settings
Baseline comparison workflow
SHAP-based model interpretation
GBM cohort prediction and publication-table generation
Installation
Option 1: Conda
conda env create -f environment.yml
conda activate seqqc-former
Option 2: pip
pip install -r requirements.txt
Configuration

Before running the pipeline, edit:

rep_error_project/config.yaml

Update the paths for your local system, including:

reference genome FASTA
tumor BAM files
normal BAM files
candidate SNV files
mappability BigWig files
replication timing BigWig files
output directories

Do not commit private paths, protected patient data, raw BAM/FASTQ files, or confidential genomic datasets.

Workflow

Run the pipeline from inside the project folder:
Example workflow:

python src/01_make_negatives_from_hc_regions.py
python src/02_merge_and_label.py
python src/03_qc_make_sites_and_windows.py
python src/04_extract_real_qc_from_bam_paired.py
python src/05_compute_map_rtim_for_all_sites.py
python src/06_extract_sequences.py
python src/07_make_features_qc.py
python src/08_create_chrom_split.py
python src/09_A_train_lightning.py
python src/10_eval_test_only.py
python src/11_robustness_test_only.py
python src/12_prepare_gbm_maf_sites.py
python src/12_extract_gbm_features.py
python src/13__extract_gbm_real_qc_from_bams.py
python src/14_predict_gbm_rep_errors.py
python src/15_predict_gbm_rep_errors_clip3_sensitivity.py
python src/16_summarize_gbm_publication_tables.py
python src/17_analyze_gbm_results.py
Results

Final public results are stored in:

rep_error_project/final_publication_results/

Representative outputs include:

ROC curves
validation ROC curve
SHAP summary plot
optimized threshold file
model performance tables
robustness analysis results
baseline comparison results
GBM prediction summaries
publication-ready summary tables

Current threshold summary:

threshold = 0.75
F1 = 0.9421841541755889
AUROC = 0.9998529146130622
AUPRC = 0.9815108017978832
Baseline Comparison

A baseline comparison script is included:

bash rep_error_project/run_baseline_comparison.sh

Before running it, update all local paths inside the script.

Data Availability

This repository does not include protected genomic data, raw sequencing data, patient-identifiable information, or large private files.

Users should provide their own input data and update the configuration file accordingly.

Large files such as BAM, FASTQ, VCF, model checkpoint, HDF5, and NPZ files should be stored outside GitHub or handled with Git LFS if appropriate.
Citation

If you use this project, please cite:

@misc{zubair2026seqqcformer,
  title={SeqQC-Former: GPU-Accelerated Detection and Analysis of Replication-Error-like Somatic Variant Artifacts},
  author={Muhammad Zubair},
  year={2026},
  url={https://github.com/Zubair11122/SeqQC-Former}
}
License

This project is released under the MIT License. See the LICENSE file for details.

Author

Muhammad Zubair

GitHub: https://github.com/Zubair11122


---

# 2. New `LICENSE`

Create a file named exactly:

```text
LICENSE
