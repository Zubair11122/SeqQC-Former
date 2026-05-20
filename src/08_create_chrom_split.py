#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import h5py
import sys

# ---------------- Paths ----------------
root = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/data_root")
h5_path = root / "features.h5"
splits_path = root / "splits.csv"

# Check if files exist
if not h5_path.exists():
    print(f"❌ Error: {h5_path} not found!")
    sys.exit(1)

print("Loading labels from HDF5...")
try:
    with h5py.File(h5_path, "r") as f:
        # Use the correct dataset names
        y = f["labels"][:]
        print(f"✅ Loaded labels with shape: {y.shape}")
        print(f"   Dataset names available: {list(f.keys())}")
        
except Exception as e:
    print(f"❌ Error loading HDF5: {e}")
    sys.exit(1)

print(f"\n📊 Dataset Statistics:")
print(f"  Total samples: {len(y):,}")
print(f"  Unique values: {np.unique(y)}")
print(f"  Positive samples (class 1): {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")
print(f"  Negative samples (class 0): {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")

# Check for extreme imbalance
if (y==1).sum() < 10 or (y==0).sum() < 10:
    print(f"⚠️ Warning: Very few samples in one class! Consider collecting more data.")

# Use random split (80/20 for train+val/test)
train_val_size = int(0.8 * len(y))
test_size = len(y) - train_val_size

print(f"\n📁 Split Configuration:")
print(f"  Train+Val size: {train_val_size:,} (80%)")
print(f"  Test size: {test_size:,} (20%)")

# First split: train_val vs test
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
train_val_idx, test_idx = next(sss1.split(np.zeros(len(y)), y))

# Second split: train vs val from train_val (80% of train_val for train, 20% for val)
y_train_val = y[train_val_idx]
train_size = int(0.8 * len(train_val_idx))
val_size = len(train_val_idx) - train_size

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
train_idx_from_val, val_idx_from_val = next(sss2.split(np.zeros(len(y_train_val)), y_train_val))

# Map back to original indices
train_idx = train_val_idx[train_idx_from_val]
val_idx = train_val_idx[val_idx_from_val]

# Verify splits are disjoint
assert len(set(train_idx) & set(val_idx)) == 0, "Train and val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train and test overlap!"
assert len(set(val_idx) & set(test_idx)) == 0, "Val and test overlap!"

print(f"\n📊 Split Details:")
print(f"  Train: {len(train_idx):,} samples")
print(f"    - Positive: {np.sum(y[train_idx]==1):,} ({np.mean(y[train_idx]==1)*100:.2f}%)")
print(f"    - Negative: {np.sum(y[train_idx]==0):,} ({np.mean(y[train_idx]==0)*100:.2f}%)")
print(f"  Validation: {len(val_idx):,} samples")
print(f"    - Positive: {np.sum(y[val_idx]==1):,} ({np.mean(y[val_idx]==1)*100:.2f}%)")
print(f"    - Negative: {np.sum(y[val_idx]==0):,} ({np.mean(y[val_idx]==0)*100:.2f}%)")
print(f"  Test: {len(test_idx):,} samples")
print(f"    - Positive: {np.sum(y[test_idx]==1):,} ({np.mean(y[test_idx]==1)*100:.2f}%)")
print(f"    - Negative: {np.sum(y[test_idx]==0):,} ({np.mean(y[test_idx]==0)*100:.2f}%)")

# Save splits
splits = pd.DataFrame({"index": range(len(y))})
splits["split"] = "unassigned"  # Default
splits.loc[train_idx, "split"] = "train"
splits.loc[val_idx, "split"] = "val"
splits.loc[test_idx, "split"] = "test"

# Verify no unassigned samples
unassigned = splits[splits["split"] == "unassigned"]
if len(unassigned) > 0:
    print(f"⚠️ Warning: {len(unassigned)} samples unassigned!")
    splits.loc[unassigned.index, "split"] = "test"  # Assign to test as fallback

splits.to_csv(splits_path, index=False)

print(f"\n✅ Splits saved to: {splits_path}")
print("\n📊 Split distribution:")
print(splits["split"].value_counts())

# Save split statistics
stats_path = root / "split_statistics.txt"
with open(stats_path, 'w') as f:
    f.write("Dataset Split Statistics\n")
    f.write("="*50 + "\n")
    f.write(f"Total samples: {len(y)}\n")
    f.write(f"Positive rate: {np.mean(y)*100:.2f}%\n\n")
    for split_name, idx in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
        y_split = y[idx]
        f.write(f"{split_name}:\n")
        f.write(f"  Samples: {len(idx)}\n")
        f.write(f"  Positive: {np.sum(y_split==1)} ({np.mean(y_split==1)*100:.2f}%)\n")
        f.write(f"  Negative: {np.sum(y_split==0)} ({np.mean(y_split==0)*100:.2f}%)\n\n")

print(f"✅ Statistics saved to: {stats_path}")

# Quick validation check
print("\n🔍 Validation Check:")
if np.allclose(np.mean(y[train_idx]==1), np.mean(y[val_idx]==1), atol=0.05):
    print("  ✓ Train and validation sets have similar class distribution")
else:
    print("  ⚠️ Warning: Train and validation class distributions differ significantly!")
    
if np.allclose(np.mean(y[train_idx]==1), np.mean(y[test_idx]==1), atol=0.05):
    print("  ✓ Train and test sets have similar class distribution")
else:
    print("  ⚠️ Warning: Train and test class distributions differ significantly!")