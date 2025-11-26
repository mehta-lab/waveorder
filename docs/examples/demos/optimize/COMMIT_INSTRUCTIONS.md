# Commit Instructions for WaveOrder Quickstart

## Summary

This directory now uses a minimal-commit strategy:
- **Commit**: Only source code and build scripts (~40 KB total)
- **Don't commit**: Binary data files (~9 MB total)
- **Regenerate**: Data files are created locally using `create_demo_data.py`

## Files Ready to Commit

### New/Modified Files ✅

```bash
git add .gitignore                    # 155 B  - Excludes data files
git add waveorder_quickstart.py       # 27 KB  - Jupytext source (loads from demo_data.b64)
git add create_demo_data.py           # 4.9 KB - Script to regenerate demo data
git add build_notebook.py             # 3.4 KB - Script to build .ipynb
git add README.md                     # 8.7 KB - Updated documentation
```

**Total to commit:** ~44 KB (vs. 5.1 MB with embedded data)

### Files in .gitignore ❌

These files exist locally but won't be committed:

```
demo_data.b64               # 5.0 MB - Base64-encoded demo dataset
demo_fov.npz                # 3.8 MB - Intermediate .npz file
waveorder_quickstart.ipynb  # Generated from .py
test_decode.npz             # Temporary test file
```

### Deleted Files 🗑️

```bash
git rm extract_demo_fov.py  # Old extraction script (replaced by create_demo_data.py)
```

## Key Changes Made

### 1. Branch Dependency: `variable-recon`

All installation references now use the `variable-recon` branch:

**waveorder_quickstart.py:**
```python
"git+https://github.com/mehta-lab/waveorder.git@variable-recon"
```

**README.md:**
- Colab link: `blob/variable-recon/...`
- Installation instructions reference `variable-recon`

### 2. Data Separation

**Before:** 5.1 MB embedded base64 string in `waveorder_quickstart.py`

**After:**
- Data extracted to `demo_data.b64` (not committed)
- Notebook loads from file at runtime:
  ```python
  data_file = Path(__file__).parent / "demo_data.b64"
  EMBEDDED_DATA = data_file.read_text().strip()
  ```

### 3. Build Workflow

New script `build_notebook.py` automates:
1. Check `demo_data.b64` exists
2. Check `jupytext` is installed
3. Convert `.py` → `.ipynb`

### 4. Reproducibility

Script `create_demo_data.py` documents exactly how to regenerate data:
1. Extract FOV `A/1/001007` from HPC zarr
2. Downsample (every 3rd z-plane)
3. Center crop (512×512)
4. Compress to .npz
5. Encode to base64

## Commit Commands

```bash
# Add new/modified files
git add .gitignore
git add waveorder_quickstart.py
git add create_demo_data.py
git add build_notebook.py
git add README.md

# Remove old files
git rm extract_demo_fov.py

# Commit
git commit -m "Refactor quickstart notebook for minimal commits

- Change branch dependency from quickstart-notebook to variable-recon
- Extract demo data to separate file (demo_data.b64, not committed)
- Add create_demo_data.py to regenerate data from HPC
- Add build_notebook.py to automate .ipynb generation
- Add .gitignore for data files
- Update README with developer workflow
- Reduce committed code from 5.1 MB to 44 KB"

# Verify what will be committed
git status
```

## For New Developers

When someone clones this branch:

```bash
# Step 1: Generate demo data (requires HPC access)
python create_demo_data.py

# Step 2: Build notebook
python build_notebook.py

# Step 3: Test
jupyter notebook waveorder_quickstart.ipynb
```

## For Colab Users

The notebook will fail on first run with a clear error:

```
FileNotFoundError: demo_data.b64 not found.
Run: python create_demo_data.py to generate it.
```

**Solution for Colab:** We need to either:
1. Host `demo_data.b64` externally (e.g., GitHub releases, S3)
2. Fall back to synthetic data for Colab
3. Provide instructions to upload the file

**Current status:** Works locally, needs Colab solution

## Verification Checklist

Before pushing:

- [ ] Branch references changed to `variable-recon`
- [ ] `demo_data.b64` exists locally (5.0 MB)
- [ ] `demo_data.b64` is in `.gitignore`
- [ ] `waveorder_quickstart.py` is 27 KB (not 5.1 MB)
- [ ] `build_notebook.py` runs successfully
- [ ] `waveorder_quickstart.ipynb` executes end-to-end locally
- [ ] Git status shows only ~44 KB of changes
- [ ] README documents the developer workflow

## What's Next

After committing, consider:

1. **Colab compatibility**: Add a fallback for when `demo_data.b64` is missing
2. **CI/CD**: Add a GitHub Action to build and test the notebook
3. **Releases**: When merging to main, update branch to stable release
4. **Data hosting**: Consider hosting `demo_data.b64` in GitHub Releases

## Questions?

- See `README.md` for full documentation
- Run `python create_demo_data.py --help` for data generation options
- Run `python build_notebook.py` for build automation
