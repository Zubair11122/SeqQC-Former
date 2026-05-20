#!/usr/bin/env python3
"""
Create feature dataset with QC metrics and sequences for deep learning model
Optimized for large datasets with chunked processing
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
import yaml
from sklearn.preprocessing import StandardScaler
import pickle

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def one_hot_encode_chunk(sequences_chunk, start_idx, h5_file):
    """
    One-hot encode a chunk of sequences and write directly to HDF5
    
    Args:
        sequences_chunk: List of sequences for this chunk
        start_idx: Starting index in the dataset
        h5_file: Open H5py file object
    """
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_samples = len(sequences_chunk)
    seq_len = len(sequences_chunk[0])
    
    # Create chunk array
    encoded_chunk = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
    
    # One-hot encode
    for i, seq in enumerate(sequences_chunk):
        seq_upper = seq.upper()
        for j, base in enumerate(seq_upper):
            if base in bases:
                encoded_chunk[i, j, bases[base]] = 1.0
    
    # Write to HDF5
    h5_file['sequences'][start_idx:start_idx + n_samples] = encoded_chunk
    
    return n_samples

def create_labels(df):
    """
    Create binary labels for training
    """
    # Use VAF threshold (0.05-0.5 for true positives)
    vaf = df['VAF'].fillna(0).values
    labels = ((vaf > 0.05) & (vaf < 0.5)).astype(np.int32)
    
    # Refine with normal alt count if available
    if 'normal_alt_count' in df.columns:
        normal_alt = df['normal_alt_count'].fillna(0).values
        somatic = normal_alt == 0
        labels = labels & somatic
    
    print(f"\nLabel distribution:")
    print(f"  Positive (1): {labels.sum():,}")
    print(f"  Negative (0): {len(labels) - labels.sum():,}")
    print(f"  Positive ratio: {labels.sum()/len(labels):.3f}")
    
    return labels

def prepare_qc_features(df, scaler=None, fit_scaler=False):
    """
    Prepare and scale QC features
    """
    # Define QC columns
    qc_columns = [
        'DP', 'AD', 'VAF', 'MQ', 'SB',
        'tumor_strand_bias', 'tumor_orientation_bias',
        'tumor_clipped_fraction', 'tumor_mismatch_fraction',
        'normal_alt_fraction', 'germline_support_flag'
    ]
    
    # Only use columns that exist
    available_cols = [col for col in qc_columns if col in df.columns]
    print(f"\nUsing QC features: {available_cols}")
    
    # Extract features
    qc_features = df[available_cols].values.astype(np.float32)
    
    # Handle missing values
    if np.any(np.isnan(qc_features)):
        print(f"  Filling {np.isnan(qc_features).sum()} NaN values with 0")
        qc_features = np.nan_to_num(qc_features, nan=0.0)
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        qc_features = scaler.fit_transform(qc_features)
        print(f"  Fitted scaler with {qc_features.shape[1]} features")
    elif scaler is not None:
        qc_features = scaler.transform(qc_features)
        print(f"  Applied pre-fitted scaler")
    else:
        print(f"  No scaling applied")
    
    return qc_features.astype(np.float32), scaler, available_cols

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        data_root = Path(config['data_root'])
        sequence_window = config.get('sequence_window', 250)
        chunk_size = 5000  # Process 5000 sequences at a time
    else:
        print("Config not found, using defaults")
        data_root = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/data_root")
        sequence_window = 250
        chunk_size = 5000
    
    # Paths
    qc_file = data_root / "qc_readlevel.csv"
    seq_file = data_root / "sequences.txt"
    output_file = data_root / "features.h5"
    scaler_file = data_root / "qc_scaler.pkl"
    
    # Check if files exist
    if not qc_file.exists():
        print(f"ERROR: QC file not found: {qc_file}")
        sys.exit(1)
    
    if not seq_file.exists():
        print(f"ERROR: Sequences file not found: {seq_file}")
        print("Please run src/extract_sequences.py first")
        sys.exit(1)
    
    # Load QC data
    print(f"\n{'='*60}")
    print("Loading QC data")
    print(f"{'='*60}")
    df = pd.read_csv(qc_file)
    print(f"Loaded {len(df):,} variants")
    
    # Load sequences
    print(f"\n{'='*60}")
    print("Loading sequences")
    print(f"{'='*60}")
    with open(seq_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]
    
    assert len(sequences) == len(df), f"Mismatch: {len(sequences)} sequences vs {len(df)} variants"
    print(f"Loaded {len(sequences):,} sequences")
    print(f"Sequence length: {len(sequences[0])}bp")
    
    # Create labels
    print(f"\n{'='*60}")
    print("Creating labels")
    print(f"{'='*60}")
    labels = create_labels(df)
    
    # Prepare QC features
    print(f"\n{'='*60}")
    print("Preparing QC features")
    print(f"{'='*60}")
    qc_features, scaler, qc_columns = prepare_qc_features(df, fit_scaler=True)
    
    # Save scaler for later use
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_file}")
    
    # Create HDF5 file with chunked storage
    print(f"\n{'='*60}")
    print(f"Creating HDF5 file: {output_file}")
    print(f"{'='*60}")
    
    n_samples = len(sequences)
    seq_length = len(sequences[0])
    n_qc_features = qc_features.shape[1]
    
    # Calculate memory requirements
    seq_memory = n_samples * seq_length * 4 * 4 / (1024**3)  # 4 bytes per float
    qc_memory = n_samples * n_qc_features * 4 / (1024**3)
    print(f"Estimated memory for sequences: {seq_memory:.2f} GB")
    print(f"Estimated memory for QC features: {qc_memory:.2f} GB")
    print(f"Processing in chunks of {chunk_size} sequences")
    
    # Create HDF5 file with datasets
    with h5py.File(output_file, 'w') as hf:
        # Create datasets with chunking for efficient I/O
        hf.create_dataset('sequences', 
                         shape=(n_samples, seq_length, 4),
                         dtype=np.float32,
                         chunks=(min(chunk_size, n_samples), seq_length, 4),
                         compression='gzip',
                         compression_opts=6)
        
        hf.create_dataset('qc_features', 
                         data=qc_features,
                         dtype=np.float32,
                         chunks=True,
                         compression='gzip',
                         compression_opts=6)
        
        hf.create_dataset('labels', 
                         data=labels,
                         dtype=np.int32,
                         chunks=True,
                         compression='gzip',
                         compression_opts=6)
        
        # Process sequences in chunks
        print("\nEncoding sequences in chunks...")
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            
            # Get chunk of sequences
            sequences_chunk = sequences[start_idx:end_idx]
            
            # One-hot encode and write to HDF5
            one_hot_encode_chunk(sequences_chunk, start_idx, hf)
        
        # Add metadata as attributes
        hf.attrs['n_samples'] = n_samples
        hf.attrs['seq_length'] = seq_length
        hf.attrs['n_qc_features'] = n_qc_features
        hf.attrs['qc_column_names'] = qc_columns  # Store as list of strings directly
        hf.attrs['n_positive'] = int(labels.sum())
        hf.attrs['n_negative'] = int(len(labels) - labels.sum())
        hf.attrs['positive_ratio'] = labels.sum() / len(labels)
        hf.attrs['label_source'] = 'VAF_threshold'
        hf.attrs['sequence_window'] = sequence_window
        
        # Store config info
        if config_path.exists():
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    try:
                        hf.attrs[f'config_{key}'] = value
                    except:
                        pass
    
    # Verify and display summary
    print(f"\n{'='*60}")
    print("✅ Successfully created dataset")
    print(f"{'='*60}")
    
    with h5py.File(output_file, 'r') as hf:
        print(f"\nDataset Summary:")
        print(f"  File: {output_file}")
        print(f"  Samples: {hf['sequences'].shape[0]:,}")
        print(f"  Sequence shape: {hf['sequences'].shape}")
        print(f"  QC features shape: {hf['qc_features'].shape}")
        print(f"  Labels shape: {hf['labels'].shape}")
        print(f"\nDataset Statistics:")
        print(f"  Positive samples: {hf.attrs['n_positive']:,}")
        print(f"  Negative samples: {hf.attrs['n_negative']:,}")
        print(f"  Positive ratio: {hf.attrs['positive_ratio']:.3f}")
        print(f"\nQC Features ({hf.attrs['n_qc_features']}):")
        # qc_column_names are now stored as strings directly
        for col in hf.attrs['qc_column_names']:
            print(f"    - {col}")
        
        # File size
        file_size = Path(output_file).stat().st_size / (1024**3)
        print(f"\nFile size: {file_size:.2f} GB")
    
    print(f"\n✅ All done! Ready for training.")
    print(f"   You can now use this file for training your model.")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Create feature dataset with QC metrics and sequences for deep learning model
Optimized for large datasets with chunked processing
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
import yaml
from sklearn.preprocessing import StandardScaler
import pickle

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def one_hot_encode_chunk(sequences_chunk, start_idx, h5_file):
    """
    One-hot encode a chunk of sequences and write directly to HDF5
    
    Args:
        sequences_chunk: List of sequences for this chunk
        start_idx: Starting index in the dataset
        h5_file: Open H5py file object
    """
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_samples = len(sequences_chunk)
    seq_len = len(sequences_chunk[0])
    
    # Create chunk array
    encoded_chunk = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
    
    # One-hot encode
    for i, seq in enumerate(sequences_chunk):
        seq_upper = seq.upper()
        for j, base in enumerate(seq_upper):
            if base in bases:
                encoded_chunk[i, j, bases[base]] = 1.0
    
    # Write to HDF5
    h5_file['sequences'][start_idx:start_idx + n_samples] = encoded_chunk
    
    return n_samples

def create_labels(df):
    """
    Create binary labels for training
    """
    # Use VAF threshold (0.05-0.5 for true positives)
    vaf = df['VAF'].fillna(0).values
    labels = ((vaf > 0.05) & (vaf < 0.5)).astype(np.int32)
    
    # Refine with normal alt count if available
    if 'normal_alt_count' in df.columns:
        normal_alt = df['normal_alt_count'].fillna(0).values
        somatic = normal_alt == 0
        labels = labels & somatic
    
    print(f"\nLabel distribution:")
    print(f"  Positive (1): {labels.sum():,}")
    print(f"  Negative (0): {len(labels) - labels.sum():,}")
    print(f"  Positive ratio: {labels.sum()/len(labels):.3f}")
    
    return labels

def prepare_qc_features(df, scaler=None, fit_scaler=False):
    """
    Prepare and scale QC features
    """
    # Define QC columns
    qc_columns = [
        'DP', 'AD', 'VAF', 'MQ', 'SB',
        'tumor_strand_bias', 'tumor_orientation_bias',
        'tumor_clipped_fraction', 'tumor_mismatch_fraction',
        'normal_alt_fraction', 'germline_support_flag'
    ]
    
    # Only use columns that exist
    available_cols = [col for col in qc_columns if col in df.columns]
    print(f"\nUsing QC features: {available_cols}")
    
    # Extract features
    qc_features = df[available_cols].values.astype(np.float32)
    
    # Handle missing values
    if np.any(np.isnan(qc_features)):
        print(f"  Filling {np.isnan(qc_features).sum()} NaN values with 0")
        qc_features = np.nan_to_num(qc_features, nan=0.0)
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        qc_features = scaler.fit_transform(qc_features)
        print(f"  Fitted scaler with {qc_features.shape[1]} features")
    elif scaler is not None:
        qc_features = scaler.transform(qc_features)
        print(f"  Applied pre-fitted scaler")
    else:
        print(f"  No scaling applied")
    
    return qc_features.astype(np.float32), scaler, available_cols

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        data_root = Path(config['data_root'])
        sequence_window = config.get('sequence_window', 250)
        chunk_size = 5000  # Process 5000 sequences at a time
    else:
        print("Config not found, using defaults")
        data_root = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/data_root")
        sequence_window = 250
        chunk_size = 5000
    
    # Paths
    qc_file = data_root / "qc_readlevel.csv"
    seq_file = data_root / "sequences.txt"
    output_file = data_root / "features.h5"
    scaler_file = data_root / "qc_scaler.pkl"
    
    # Check if files exist
    if not qc_file.exists():
        print(f"ERROR: QC file not found: {qc_file}")
        sys.exit(1)
    
    if not seq_file.exists():
        print(f"ERROR: Sequences file not found: {seq_file}")
        print("Please run src/extract_sequences.py first")
        sys.exit(1)
    
    # Load QC data
    print(f"\n{'='*60}")
    print("Loading QC data")
    print(f"{'='*60}")
    df = pd.read_csv(qc_file)
    print(f"Loaded {len(df):,} variants")
    
    # Load sequences
    print(f"\n{'='*60}")
    print("Loading sequences")
    print(f"{'='*60}")
    with open(seq_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]
    
    assert len(sequences) == len(df), f"Mismatch: {len(sequences)} sequences vs {len(df)} variants"
    print(f"Loaded {len(sequences):,} sequences")
    print(f"Sequence length: {len(sequences[0])}bp")
    
    # Create labels
    print(f"\n{'='*60}")
    print("Creating labels")
    print(f"{'='*60}")
    labels = create_labels(df)
    
    # Prepare QC features
    print(f"\n{'='*60}")
    print("Preparing QC features")
    print(f"{'='*60}")
    qc_features, scaler, qc_columns = prepare_qc_features(df, fit_scaler=True)
    
    # Save scaler for later use
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_file}")
    
    # Create HDF5 file with chunked storage
    print(f"\n{'='*60}")
    print(f"Creating HDF5 file: {output_file}")
    print(f"{'='*60}")
    
    n_samples = len(sequences)
    seq_length = len(sequences[0])
    n_qc_features = qc_features.shape[1]
    
    # Calculate memory requirements
    seq_memory = n_samples * seq_length * 4 * 4 / (1024**3)  # 4 bytes per float
    qc_memory = n_samples * n_qc_features * 4 / (1024**3)
    print(f"Estimated memory for sequences: {seq_memory:.2f} GB")
    print(f"Estimated memory for QC features: {qc_memory:.2f} GB")
    print(f"Processing in chunks of {chunk_size} sequences")
    
    # Create HDF5 file with datasets
    with h5py.File(output_file, 'w') as hf:
        # Create datasets with chunking for efficient I/O
        hf.create_dataset('sequences', 
                         shape=(n_samples, seq_length, 4),
                         dtype=np.float32,
                         chunks=(min(chunk_size, n_samples), seq_length, 4),
                         compression='gzip',
                         compression_opts=6)
        
        hf.create_dataset('qc_features', 
                         data=qc_features,
                         dtype=np.float32,
                         chunks=True,
                         compression='gzip',
                         compression_opts=6)
        
        hf.create_dataset('labels', 
                         data=labels,
                         dtype=np.int32,
                         chunks=True,
                         compression='gzip',
                         compression_opts=6)
        
        # Process sequences in chunks
        print("\nEncoding sequences in chunks...")
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            
            # Get chunk of sequences
            sequences_chunk = sequences[start_idx:end_idx]
            
            # One-hot encode and write to HDF5
            one_hot_encode_chunk(sequences_chunk, start_idx, hf)
        
        # Add metadata as attributes
        hf.attrs['n_samples'] = n_samples
        hf.attrs['seq_length'] = seq_length
        hf.attrs['n_qc_features'] = n_qc_features
        hf.attrs['qc_column_names'] = qc_columns  # Store as list of strings directly
        hf.attrs['n_positive'] = int(labels.sum())
        hf.attrs['n_negative'] = int(len(labels) - labels.sum())
        hf.attrs['positive_ratio'] = labels.sum() / len(labels)
        hf.attrs['label_source'] = 'VAF_threshold'
        hf.attrs['sequence_window'] = sequence_window
        
        # Store config info
        if config_path.exists():
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    try:
                        hf.attrs[f'config_{key}'] = value
                    except:
                        pass
    
    # Verify and display summary
    print(f"\n{'='*60}")
    print("✅ Successfully created dataset")
    print(f"{'='*60}")
    
    with h5py.File(output_file, 'r') as hf:
        print(f"\nDataset Summary:")
        print(f"  File: {output_file}")
        print(f"  Samples: {hf['sequences'].shape[0]:,}")
        print(f"  Sequence shape: {hf['sequences'].shape}")
        print(f"  QC features shape: {hf['qc_features'].shape}")
        print(f"  Labels shape: {hf['labels'].shape}")
        print(f"\nDataset Statistics:")
        print(f"  Positive samples: {hf.attrs['n_positive']:,}")
        print(f"  Negative samples: {hf.attrs['n_negative']:,}")
        print(f"  Positive ratio: {hf.attrs['positive_ratio']:.3f}")
        print(f"\nQC Features ({hf.attrs['n_qc_features']}):")
        # qc_column_names are now stored as strings directly
        for col in hf.attrs['qc_column_names']:
            print(f"    - {col}")
        
        # File size
        file_size = Path(output_file).stat().st_size / (1024**3)
        print(f"\nFile size: {file_size:.2f} GB")
    
    print(f"\n✅ All done! Ready for training.")
    print(f"   You can now use this file for training your model.")

if __name__ == "__main__":
    main()