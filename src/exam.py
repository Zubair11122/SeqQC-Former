#!/usr/bin/env python3
import h5py
from pathlib import Path

root = Path("/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/GPU/data_root")
h5_path = root / "features.h5"

print(f"Checking file: {h5_path}")
print(f"File exists: {h5_path.exists()}")

if h5_path.exists():
    with h5py.File(h5_path, "r") as f:
        print("\n📁 Available datasets and groups:")
        
        def print_structure(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name} - Shape: {obj.shape} - Dtype: {obj.dtype}")
            else:
                print(f"{indent}📁 Group: {name}")
        
        f.visititems(print_structure)
        
        print("\n🔍 Quick check:")
        for key in f.keys():
            print(f"  - {key}")
else:
    print("❌ File not found!")