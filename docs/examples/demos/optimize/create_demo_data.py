#!/usr/bin/env python
"""
Create demo_data.b64 for the WaveOrder optimization quickstart notebook.

This script extracts a small FOV from the HPC zarr store and encodes it as base64.
The resulting demo_data.b64 file is loaded by the notebook at runtime.

Usage:
    python create_demo_data.py

Requirements:
    - Access to HPC data
    - iohub installed
    - numpy installed

The script will:
    1. Extract FOV from HPC zarr store
    2. Save as compressed .npz
    3. Encode to base64
    4. Save as demo_data.b64
"""

import base64
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr


def extract_fov_from_hpc():
    """Extract a small FOV from HPC zarr store"""

    # HPC data location
    hpc_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/0-convert/live_imaging/tracking_symlink.zarr"
    fov_name = "A/1/001007"

    print(f"Opening zarr store: {hpc_path}")
    store = open_ome_zarr(hpc_path)

    print(f"Extracting FOV: {fov_name}")
    fov = store[fov_name]

    # Extract a subset to keep file size small
    # Time=0, Channel=0, Z subset, Y/X center crop
    time_idx = 0
    channel_idx = 0
    z_slice = slice(None, None, 3)  # Every 3rd z-plane
    y_slice = slice(256, 768)  # 512 pixels
    x_slice = slice(256, 768)  # 512 pixels

    zyx_data = fov.data[time_idx][channel_idx][z_slice, y_slice, x_slice]
    scale = fov.scale

    print(f"  Extracted shape: {zyx_data.shape}")
    print(f"  Scale: {scale}")

    return zyx_data, scale


def save_as_npz(zyx_data, scale):
    """Save extracted data as compressed .npz"""

    demo_file = Path("demo_fov.npz")

    np.savez_compressed(
        demo_file,
        data=zyx_data,
        scale=scale,
        fov_name="A/1/001007",
        channel_idx=0,
        time_idx=0,
        subset_info={
            'z_slice': 'slice(None, None, 3)',
            'y_slice': 'slice(256, 768)',
            'x_slice': 'slice(256, 768)',
        }
    )

    file_size_mb = demo_file.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved to {demo_file}")
    print(f"  Size: {file_size_mb:.2f} MB")

    return demo_file


def encode_to_base64(npz_file):
    """Encode .npz file to base64"""

    print(f"\nEncoding {npz_file} to base64...")

    with open(npz_file, 'rb') as f:
        data_bytes = f.read()

    encoded = base64.b64encode(data_bytes).decode('utf-8')

    # Save to demo_data.b64
    b64_file = Path("demo_data.b64")
    b64_file.write_text(encoded)

    encoded_size_mb = len(encoded) / 1024 / 1024
    print(f"✓ Saved to {b64_file}")
    print(f"  Size: {encoded_size_mb:.2f} MB")

    return b64_file


def verify_encoding(b64_file):
    """Verify the base64 encoding works"""

    print(f"\nVerifying {b64_file}...")

    # Read back and decode
    encoded = b64_file.read_text().strip()
    decoded = base64.b64decode(encoded)

    # Write to temp file
    temp_file = Path("test_decode.npz")
    temp_file.write_bytes(decoded)

    # Try to load with numpy
    data = np.load(temp_file)
    print(f"✓ Successfully decoded and loaded")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Data shape: {data['data'].shape}")
    print(f"  Scale: {data['scale']}")

    # Clean up
    temp_file.unlink()

    return True


def main():
    """Main workflow"""

    print("=" * 80)
    print("WaveOrder Demo Data Creation")
    print("=" * 80)
    print()

    try:
        # Step 1: Extract from HPC
        print("Step 1: Extract FOV from HPC")
        print("-" * 80)
        zyx_data, scale = extract_fov_from_hpc()

        # Step 2: Save as .npz
        print("\nStep 2: Save as compressed .npz")
        print("-" * 80)
        npz_file = save_as_npz(zyx_data, scale)

        # Step 3: Encode to base64
        print("\nStep 3: Encode to base64")
        print("-" * 80)
        b64_file = encode_to_base64(npz_file)

        # Step 4: Verify
        print("\nStep 4: Verify encoding")
        print("-" * 80)
        verify_encoding(b64_file)

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\nCreated files:")
        print(f"  • demo_fov.npz - Intermediate .npz file")
        print(f"  • demo_data.b64 - Base64-encoded data for notebook")
        print(f"\nNext steps:")
        print(f"  1. The notebook will automatically load demo_data.b64")
        print(f"  2. You can delete demo_fov.npz (it's just intermediate)")
        print(f"  3. Add demo_data.b64 to .gitignore (don't commit the data)")
        print(f"  4. Convert notebook: jupytext --to ipynb waveorder_quickstart.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure you're on HPC or have access to the data path")
        print("  • Ensure iohub is installed: pip install iohub")
        print("  • Check that the zarr store path is correct")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
