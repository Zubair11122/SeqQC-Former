SeqQC-Former
SeqQC-Former is a GPU-accelerated deep-learning framework for QC-aware review prioritization of candidate somatic single-nucleotide variants (SNVs) in cancer genomics. The framework integrates local nucleotide sequence context with read-level quality-control (QC) covariates derived from matched tumor-normal sequencing data to generate prioritization scores that aid downstream variant review.
This repository contains the implementation of the SeqQC-Former framework as described in the manuscript:
"SeqQC-Former: A Sequence–Quality Fusion Framework for QC-Aware Review Prioritization of Candidate Somatic SNVs in Cancer Genomics"
Overview
Accurate prioritization of somatic SNVs remains challenging due to substantial variability in sequencing quality across genomic loci. SeqQC-Former is designed as a post-calling prioritization tool that:
•	Integrates a 501-bp local nucleotide context with structured QC covariates
•	Generates QC-aware review prioritization scores for candidate somatic SNVs
•	Supports downstream review and prioritization of candidate loci requiring further investigation
•	Operates as a complementary layer to existing somatic variant callers rather than replacing them
Important: SeqQC-Former produces QC-dependent prioritization scores and is not intended to infer biological truth, replace somatic variant callers, or offer experimentally validated variant classifications.
Key Features
•	Sequence–Quality Fusion Architecture: Combines a convolutional-transformer sequence encoder with an MLP-based QC encoder
•	Comprehensive QC Feature Set: 11 read-level QC covariates from matched tumor-normal data
•	SEQC2-Derived Benchmarking: Evaluated on 89,447 candidate loci (1,378 positive, 88,069 negative)
•	Chromosome-Held-Out Validation: Minimizes genomic position leakage for robust generalization assessment
•	External Application: Applied to 53,164 glioblastoma somatic SNVs as a domain-shift case study
•	Model Interpretability: SHAP-based feature attribution for QC branch analysis
•	Comprehensive Analyses: Ablation studies, robustness testing, calibration analysis, and baseline comparisons
Repository Structure
text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
└── seqqc_formER/
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
Main Components
Data Processing
•	SEQC2-derived candidate locus construction from truth-set somatic SNVs
•	501-bp one-hot encoded sequence window extraction
•	11 read-level QC covariate extraction from tumor-normal BAM files
•	HDF5 feature assembly for efficient storage
Model Architecture
•	Sequence Branch: 4 × 501 one-hot encoding → 3 Conv1D layers (4→32→64→64) → Adaptive pooling → 2-layer Transformer encoder (4 heads, d_model=64)
•	QC Branch: 11-dimensional input → MLP (11→32→16) with batch normalization and ReLU
•	Fusion: Concatenation of 64-dim sequence and 16-dim QC embeddings → Classifier (80→64→32→1)
•	Output: Sigmoid-transformed QC-aware review prioritization score
Training and Evaluation
•	Weighted binary cross-entropy loss for class imbalance
•	AdamW optimizer with learning rate 1×10⁻⁴
•	Stratified and chromosome-held-out evaluation splits
•	Bootstrap resampling (1,000 iterations) for confidence intervals
•	DeLong's test for AUROC comparisons
Installation
Option 1: Conda
bash
conda env create -f environment.yml
conda activate seqqc-former
Option 2: pip
bash
pip install -r requirements.txt
Configuration
Edit seqqc_formER/config.yaml with your local paths:
•	Reference genome FASTA
•	Tumor and normal BAM files
•	Candidate SNV files
•	Output directories
•	Model hyperparameters
Note: Do not commit private paths, protected patient data, raw BAM/FASTQ files, or confidential genomic datasets.
Workflow
Run the pipeline sequentially from inside the project folder:
bash
# Data preparation
python src/01_make_negatives_from_hc_regions.py
python src/02_merge_and_label.py
python src/03_qc_make_sites_and_windows.py
python src/04_extract_real_qc_from_bam_paired.py
python src/05_compute_map_rtim_for_all_sites.py
python src/06_extract_sequences.py
python src/07_make_features_qc.py
python src/08_create_chrom_split.py

# Model training and evaluation
python src/09_A_train_lightning.py
python src/10_eval_test_only.py

# Analyses
python src/11_robustness_test_only.py

# External GBM application
python src/12_prepare_gbm_maf_sites.py
python src/12_extract_gbm_features.py
python src/13_extract_gbm_real_qc_from_bams.py
python src/14_predict_gbm_rep_errors.py
python src/15_predict_gbm_rep_errors_clip3_sensitivity.py

# Results summarization
python src/16_summarize_gbm_publication_tables.py
python src/17_analyze_gbm_results.py
Baseline Comparison
bash
bash seqqc_formER/run_baseline_comparison.sh
Note: Update all local paths inside the script before running.
Key Results
SEQC2 Held-Out Performance
Metric	Value
AUROC	0.99997
AUPRC	0.9982
F1-score (threshold=0.75)	0.9386
Precision	0.8932
Recall	0.9891
Chromosome-Held-Out Validation
Metric	Value
AUROC	0.9479
AUPRC	0.9448
F1-score (threshold=0.95)	0.8596
Note: Performance metrics reflect QC-aware prioritization capability rather than independent biological variant correctness validation.
Model Interpretability
•	SHAP analysis identifies allele-support and normal-evidence covariates as the most influential predictors
•	Mapping quality and strand bias features provide additional predictive signal
•	QC features dominate predictive signal under current SEQC2-derived labeling
External Application
SeqQC-Former was applied to 53,164 glioblastoma somatic SNVs:
Configuration	Prioritized Variants	Percentage
Default (clip1)	30,177	56.76%
Stringent (clip3)	3,715	6.99%
Note: Results are sensitivity-dependent and should be interpreted as exploratory prioritization under domain-shift conditions.
Data Availability
This repository does not include:
•	Protected genomic data
•	Raw sequencing data
•	Patient-identifiable information
•	Large private files (BAM, FASTQ, VCF, large checkpoint files)
Users must provide their own input data and update configuration accordingly.
Large files should be stored outside GitHub or handled with Git LFS where appropriate.
Citation
If you use this project, please cite:
text
@article{zubair2026seqqcformer,
  title={SeqQC-Former: A Sequence–Quality Fusion Framework for QC-Aware Review Prioritization of Candidate Somatic SNVs in Cancer Genomics},
  author={Zubair, Muhammad and Li, Jianqiang and Qian, Jun and Wang, Zitong},
  journal={Computational Biology and Chemistry},
  year={2026}
}
Also cite:
text
@misc{zubair2026seqqcformer_code,
  title={SeqQC-Former: Sequence–Quality Fusion Framework for QC-Aware Review Prioritization},
  author={Zubair, Muhammad},
  year={2026},
  url={https://github.com/Zubair11122/SeqQC-Former}
}
License
This project is released under the MIT License. See the LICENSE file for details.
Author
Muhammad Zubair
GitHub: https://github.com/Zubair11122
Acknowledgments
This work was supported by the Beijing Natural Science Foundation - Haidian Original Innovation Joint Fund (project no. L252088).
The authors acknowledge the use of public data resources including TCGA and SEQC2 benchmarking materials.

