#!/usr/bin/env python3
"""
VCF to CSV Converter for Somatic Variant Analysis
- Handles MuTect2 (muTect2/mutect2) and Strelka2 VCFs
- Processes compressed (.gz) and uncompressed VCFs
- Auto-creates output directory
- Comprehensive error logging
"""

import argparse
import gzip
import re
import pandas as pd
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vcf_conversion.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_path(path: Path, is_dir: bool = False) -> Path:
    """Validate input/output paths."""
    try:
        if is_dir:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.parent.exists():
            path.parent.mkdir(parents=True)
        return path.resolve()
    except Exception as e:
        logger.error(f"Path validation failed for {path}: {str(e)}")
        raise


def open_vcf(vcf_path: Path):
    """Open VCF file (handles both gzipped and uncompressed)."""
    try:
        if vcf_path.suffix == '.gz':
            return gzip.open(vcf_path, 'rt')
        return open(vcf_path, 'r')
    except Exception as e:
        logger.error(f"Failed to open VCF file {vcf_path}: {str(e)}")
        raise


def parse_vcf(vcf_path: Path) -> list:
    """Parse VCF and extract variant information."""
    variants = []
    try:
        with open_vcf(vcf_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                try:
                    fields = line.strip().split('\t')
                    chrom = fields[0].replace('chr', '')
                    pos = int(fields[1])
                    ref = fields[3]
                    alts = fields[4].split(',')
                    qual = fields[5]
                    info = fields[7]

                    # Extract AF from INFO or use QUAL
                    af_match = re.search(r'AF=([0-9\.eE+-]+)', info)
                    score = float(af_match.group(1)) if af_match else float(qual) if qual != '.' else 0.0

                    for alt in alts:
                        variants.append({
                            'chrom': chrom,
                            'pos': pos,
                            'ref': ref,
                            'alt': alt,
                            'score': score
                        })

                except Exception as e:
                    logger.warning(f"Skipping malformed line: {line.strip()}. Error: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"VCF parsing failed: {str(e)}")
        raise

    return variants


def get_vcf_path(input_dir: Path, sample: str, caller: str) -> Path:
    """Handle case variations in VCF filenames."""
    base_name = f"WES_{sample}.bwa"

    patterns = {
        'Mutect2': [
            f"{base_name}.muTect2.vcf.gz",  # Your actual filename
            f"{base_name}.mutect2.vcf.gz",
            f"{base_name}.muTect2.vcf"
        ],
        'Strelka2': [
            f"{base_name}.strelka.vcf.gz",  # Your actual filename
            f"{base_name}.strelka2.vcf.gz",
            f"{base_name}.strelka.vcf"
        ]
    }

    for pattern in patterns[caller]:
        vcf_path = input_dir / pattern
        if vcf_path.exists():
            return vcf_path

    # Show existing files for debugging
    existing_files = list(input_dir.glob(f"{base_name}.*"))
    raise FileNotFoundError(
        f"No {caller} VCF found matching patterns: {patterns[caller]}\n"
        f"Existing files: {existing_files}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert somatic VCFs to analysis-ready CSV format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sample', required=True, choices=['IL_1', 'FD_1'], help='Sample identifier')
    parser.add_argument('--caller', required=True, choices=['Mutect2', 'Strelka2'], help='Variant caller used')
    parser.add_argument('--input-dir', default='/home/zubair/Project/data_root/baseline_vcfs',
                        help='Input VCF directory')
    parser.add_argument('--output-dir', default='/home/zubair/Project/data_root/baseline_out',
                        help='Output CSV directory')

    args = parser.parse_args()

    try:
        # Configure paths
        input_dir = validate_path(Path(args.input_dir), is_dir=True)
        output_dir = validate_path(Path(args.output_dir), is_dir=True)

        # Get case-sensitive VCF path
        vcf_path = get_vcf_path(input_dir, args.sample, args.caller)
        output_csv = output_dir / f"{args.caller.lower()}_{args.sample}_bwa.csv"

        logger.info(f"Processing {vcf_path}...")
        variants = parse_vcf(vcf_path)

        if not variants:
            logger.warning("No variants found in VCF!")
        else:
            pd.DataFrame(variants).to_csv(output_csv, index=False)
            logger.info(f"Successfully wrote {len(variants)} variants to {output_csv}")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()