#!/usr/bin/env python3
"""
Simple sequence extractor - adds "chr" prefix to match reference
"""

import pandas as pd
import pysam
from pathlib import Path
from tqdm import tqdm
import sys

def main():
    # Paths
    qc_file = Path("data_root/qc_readlevel.csv")
    ref_fasta = Path("reference/hg38.fa")
    output_file = Path("data_root/sequences.txt")
    flank_size = 250  # 250bp each side = 501bp total
    
    # Check if files exist
    if not qc_file.exists():
        print(f"ERROR: {qc_file} not found")
        sys.exit(1)
    
    if not ref_fasta.exists():
        print(f"ERROR: {ref_fasta} not found")
        sys.exit(1)
    
    # Load variants
    print(f"Loading variants from {qc_file}")
    df = pd.read_csv(qc_file)
    print(f"Loaded {len(df)} variants")
    
    # Parse coordinates
    print("Parsing coordinates...")
    coords = df['key'].str.split(':', expand=True)
    chroms = coords[0].tolist()
    positions = coords[1].astype(int).tolist()
    refs = coords[2].tolist()
    alts = coords[3].tolist()
    
    # Open reference
    print(f"Opening reference: {ref_fasta}")
    fasta = pysam.FastaFile(str(ref_fasta))
    
    # Extract sequences
    print(f"Extracting {len(df)} sequences (length={2*flank_size+1}bp)...")
    sequences = []
    missing = 0
    
    for i, (chrom, pos, ref) in enumerate(tqdm(zip(chroms, positions, refs), total=len(df))):
        # Add "chr" prefix if not present
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"
        
        # Calculate coordinates
        start = pos - flank_size - 1  # 0-based
        end = pos + flank_size
        
        try:
            seq = fasta.fetch(chrom, start, end).upper()
            if len(seq) != 2*flank_size + 1:
                missing += 1
                seq = 'N' * (2*flank_size + 1)
            sequences.append(seq)
        except Exception as e:
            missing += 1
            sequences.append('N' * (2*flank_size + 1))
            if missing <= 5:
                print(f"Error at {chrom}:{pos}: {e}")
    
    fasta.close()
    
    # Save sequences
    print(f"\nSaving to {output_file}")
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')
    
    print(f"\n✅ Complete!")
    print(f"   Total: {len(sequences):,}")
    print(f"   Success: {len(sequences) - missing:,}")
    print(f"   Missing: {missing:,}")
    print(f"\nSample sequences:")
    for i, seq in enumerate(sequences[:3]):
        print(f"  {i+1}: {seq[:50]}...{seq[-50:]}")

if __name__ == "__main__":
    main()