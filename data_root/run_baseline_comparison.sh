#!/bin/bash
set -e

# ==========================================
# Baseline Comparison Pipeline (Corrected)
# ==========================================

# ---------- Paths ----------
TOOLS_DIR="/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/tools"
TOOLS_VCF_DIR="/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/tools_vcf"
DATA_ROOT="/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU"
BASELINE_SCRIPT="$DATA_ROOT/src/baseline_comparison_auto.py"

# ---------- Reference & BAM files ----------
REFERENCE="$DATA_ROOT/reference/hg38.fa"
TUMOR_BAM="$DATA_ROOT/data_root/bam/WES_FD_T_1.bwa.dedup.bam"
NORMAL_BAM="$DATA_ROOT/data_root/bam/WES_FD_N_1.bwa.dedup.bam"

# ---------- Create output directory ----------
mkdir -p "$TOOLS_VCF_DIR"

# ==========================================
# Activate Conda Environment
# ==========================================
echo "Activating conda environment..."
source /home/BD-4/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/envs/lit310_gpu_v2
echo "Conda environment activated."
python --version

# ==========================================
# Make all tool scripts executable
# ==========================================
echo "Making tool scripts executable..."
find "$TOOLS_DIR" -type f -name "*.sh" -exec chmod +x {} \;
find "$TOOLS_DIR" -type f -name "*.py" -exec chmod +x {} \;

# ==========================================
# Run GATK Mutect2
# ==========================================
echo "Running GATK Mutect2..."
cd "$TOOLS_DIR/gatk"
if [ -f "gatk-package-4.6.2.0-local.jar" ]; then
    java -jar gatk-package-4.6.2.0-local.jar Mutect2 \
        -R "$REFERENCE" \
        -I "$TUMOR_BAM" \
        -I "$NORMAL_BAM" \
        -O "$TOOLS_VCF_DIR/mutect2.vcf.gz"
else
    echo "GATK jar not found"
fi

# ==========================================
# Run Strelka2
# ==========================================
echo "Running Strelka2..."
cd "$TOOLS_DIR/strelka2/strelka-2.9.10.centos6_x86_64/bin"
if [ -f "configureStrelkaSomaticWorkflow.py" ]; then
    python configureStrelkaSomaticWorkflow.py \
        --tumorBam "$TUMOR_BAM" \
        --normalBam "$NORMAL_BAM" \
        --referenceFasta "$REFERENCE" \
        --runDir strelka_run
    cd strelka_run
    python runWorkflow.py -m local -j 4
    find results -name "*.vcf.gz" -exec cp {} "$TOOLS_VCF_DIR/strelka2.vcf.gz" \;
else
    echo "Strelka2 workflow script not found"
fi

# ==========================================
# Run NeuSomatic
# ==========================================
echo "Running NeuSomatic..."
cd "$TOOLS_DIR/neusomatic/neusomatic-0.2.1/neusomatic/python"
if [ -f "call.py" ]; then
    python call.py \
        --tumor "$TUMOR_BAM" \
        --normal "$NORMAL_BAM" \
        --reference "$REFERENCE" \
        --out "$TOOLS_VCF_DIR/neusomatic.vcf.gz"
else
    echo "NeuSomatic call.py not found"
fi

# ==========================================
# Skip Lancet2 (needs compilation)
# ==========================================
echo "Skipping Lancet2 (not compiled)"

# ==========================================
# Run bcftools + samtools
# ==========================================
echo "Running samtools + bcftools..."
samtools mpileup -f "$REFERENCE" "$TUMOR_BAM" "$NORMAL_BAM" | \
bcftools call -mv -Oz -o "$TOOLS_VCF_DIR/samtools.vcf.gz"

# ==========================================
# Index all VCFs
# ==========================================
echo "Indexing all VCF files..."
cd "$TOOLS_VCF_DIR"
for f in *.vcf.gz; do
    if [ -f "$f" ]; then
        echo "Indexing $f"
        tabix -p vcf "$f"
    fi
done

# ==========================================
# Run Python baseline comparison
# ==========================================
echo "Running baseline_comparison_auto.py..."
cd "$DATA_ROOT"
python "$BASELINE_SCRIPT"

echo "=========================================="
echo "Baseline comparison completed successfully"
echo "Results saved in $TOOLS_VCF_DIR and baseline CSV."
echo "=========================================="