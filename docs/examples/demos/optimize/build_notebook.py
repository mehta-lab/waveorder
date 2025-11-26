#!/usr/bin/env python
"""
Build the waveorder_quickstart.ipynb notebook from the Jupytext source.

This script:
1. Checks that demo_data.b64 exists (required for the notebook to run)
2. Converts waveorder_quickstart.py to waveorder_quickstart.ipynb using jupytext
3. Verifies the notebook was created successfully

Usage:
    python build_notebook.py

Requirements:
    - jupytext installed: pip install jupytext
    - demo_data.b64 exists (run create_demo_data.py first)
"""

import subprocess
import sys
from pathlib import Path


def check_demo_data():
    """Check that demo_data.b64 exists"""

    demo_file = Path("demo_data.b64")

    if not demo_file.exists():
        print("✗ Error: demo_data.b64 not found")
        print()
        print("Please run this first:")
        print("  python create_demo_data.py")
        print()
        print("This will generate the demo_data.b64 file from HPC.")
        return False

    file_size_mb = demo_file.stat().st_size / 1024 / 1024
    print(f"✓ Found demo_data.b64 ({file_size_mb:.1f} MB)")
    return True


def check_jupytext():
    """Check that jupytext is installed"""

    try:
        result = subprocess.run(
            ["jupytext", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"✓ Found jupytext {version}")
        return True
    except FileNotFoundError:
        print("✗ Error: jupytext not found")
        print()
        print("Please install jupytext:")
        print("  pip install jupytext")
        return False
    except subprocess.CalledProcessError:
        print("✗ Error: Could not run jupytext")
        return False


def build_notebook():
    """Convert .py to .ipynb using jupytext"""

    source_file = Path("waveorder_quickstart.py")
    output_file = Path("waveorder_quickstart.ipynb")

    if not source_file.exists():
        print(f"✗ Error: {source_file} not found")
        return False

    print(f"\nBuilding notebook from {source_file}...")

    try:
        subprocess.run(
            ["jupytext", "--to", "ipynb", str(source_file)],
            check=True
        )

        if output_file.exists():
            file_size_kb = output_file.stat().st_size / 1024
            print(f"✓ Created {output_file} ({file_size_kb:.1f} KB)")
            return True
        else:
            print(f"✗ Error: {output_file} was not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Error: jupytext failed with code {e.returncode}")
        return False


def main():
    """Main workflow"""

    print("=" * 80)
    print("WaveOrder Quickstart Notebook Builder")
    print("=" * 80)
    print()

    print("Step 1: Checking dependencies")
    print("-" * 80)

    if not check_demo_data():
        return 1

    if not check_jupytext():
        return 1

    print("\nStep 2: Building notebook")
    print("-" * 80)

    if not build_notebook():
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test the notebook:")
    print("     jupyter notebook waveorder_quickstart.ipynb")
    print()
    print("  2. Or upload to Google Colab:")
    print("     https://colab.research.google.com/")
    print()
    print("Note: The notebook will automatically load data from demo_data.b64")

    return 0


if __name__ == "__main__":
    sys.exit(main())
