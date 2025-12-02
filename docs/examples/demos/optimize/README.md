# WaveOrder Optimization Quickstart

A self-contained notebook demonstrating WaveOrder's optimization capabilities for computational microscopy.

## Quick Start

### ⚠️ Important: Branch Dependency

**Current Status:** This notebook requires the `variable-recon` branch of WaveOrder.

The notebook installs directly from GitHub:
```python
pip install git+https://github.com/mehta-lab/waveorder.git@variable-recon
```

**When `variable-recon` merges to `main`:**
- Update the install line to use the stable release: `"waveorder>=2.0.0"`
- Do not delete the branch (keep as reference)

### Run in Google Colab

1. Upload `waveorder_quickstart.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells (~15 minutes)
3. The notebook automatically downloads demo data from CZ Biohub's public server

Or use this direct link (once pushed to GitHub):
```
https://colab.research.google.com/github/mehta-lab/waveorder/blob/variable-recon/docs/examples/demos/optimize/waveorder_quickstart.ipynb
```

### Run Locally

```bash
cd ~/waveorder/docs/examples/demos/optimize
jupyter notebook waveorder_quickstart.ipynb
```

## What This Notebook Demonstrates

### Physics-Informed Reconstruction

WaveOrder uses differentiable forward models to reconstruct quantitative phase from label-free microscopy images.

### Auto-Tuning via Optimization

The notebook optimizes 5 optical parameters:

1. **Defocus offset** (`z_offset`) - Focal plane alignment
2. **Detection NA** - Effective detection cone angle
3. **Illumination NA** - Illumination cone angle
4. **Tilt zenith** - Angle from optical axis
5. **Tilt azimuth** - Rotation around optical axis

### Mid-Band Frequency Loss

Uses mid-band frequency power as the optimization objective to maximize sharpness and focus.

### Results

Typical improvements:
- 10-30% increase in mid-band power
- Sharper features
- Better contrast
- Physically interpretable parameters

## Notebook Contents

1. **Setup** (1 cell) - Install dependencies, import libraries, define helper functions
2. **Load & Visualize Data** (1 cell) - Extract embedded data, show z-stack
3. **Initial Reconstruction** (2 cells) - Configure parameters, run reconstruction
4. **Optimize Reconstruction** (3 cells) - Configure optimization, run 20 iterations, analyze convergence
5. **Compare Results** (1 cell) - Side-by-side comparison with metrics
6. **Summary** (markdown) - Interpretation and next steps

## Using Your Own Data

The notebook includes embedded demo data for immediate use. To use your own data:

### Option 1: Load from Array (Simple)

In the data loading cell, comment out the embedded data and load your array:

```python
# Comment out these lines:
# demo_file = load_embedded_demo_data()
# zyx_data, z_scale, y_scale, x_scale = load_demo_data(demo_file)

# Load your own data:
zyx_data = your_data_array  # Shape: (Z, Y, X)
z_scale, y_scale, x_scale = 0.5, 0.108, 0.108  # Your pixel sizes in µm
```

### Option 2: Load from File

```python
import numpy as np

# Load from .npz
data = np.load('your_data.npz')
zyx_data = data['zyx']
z_scale, y_scale, x_scale = 0.5, 0.108, 0.108

# Or load from .npy
zyx_data = np.load('your_data.npy')
z_scale, y_scale, x_scale = 0.5, 0.108, 0.108
```

### Data Requirements

- **Format**: NumPy array, shape (Z, Y, X)
- **Z-slices**: 10-30 planes recommended
- **FOV**: 128-512 pixels per side
- **Data type**: Float32 or Float64
- **Scale**: Pixel sizes in micrometers

## Key Features

✅ **Self-contained** - Demo data downloads automatically from public server
✅ **Proper Jupytext format** - Clean cell structure with `# %% [markdown]`
✅ **Prints loss every iteration** - Full optimization transparency
✅ **Convergence analysis** - Recommends next steps if not converged
✅ **Grayscale visualizations** - Publication-ready figures
✅ **Central crops** - Shows middle 50% for clarity
✅ **Streamlined** - Only 8 executable cells

## Version Management

### Current (Pre-Merge)

```python
pip install git+https://github.com/mehta-lab/waveorder.git@variable-recon
```

### After Merge to Main

Update the notebook's install command to:
```python
pip install "waveorder>=2.0.0"
```

Then:
1. Test notebook with new version
2. Update Colab links from `blob/variable-recon/` to `blob/main/`
3. Keep the branch (don't delete it)

## Troubleshooting

### Optimization not converging

- Increase `NUM_ITERATIONS` to 50 or 100
- Reduce learning rates in `OPTIMIZABLE_PARAMS` (try halving them)
- Run the optimization cell again to continue from current parameters

### Out of memory

- Use a smaller FOV (fewer pixels or z-slices)
- Close other Colab notebooks
- Restart runtime and try again

### Poor reconstruction quality

- Verify wavelength matches your microscope
- Check pixel sizes are correct
- Adjust regularization strength
- Ensure initial parameter guesses are reasonable

## For Developers

### Repository Structure

This directory contains:

**Files to commit:**
- `waveorder_quickstart.py` (~27 KB) - Jupytext source
- `README.md` - Documentation
- `.gitignore` - Excludes downloaded data

**Files NOT to commit (in .gitignore):**
- `waveorder/20x.zarr` (~123 MB) - Downloaded demo dataset
- `waveorder_quickstart.ipynb` - Generated notebook file

### Developer Workflow

#### Making Changes

1. **Edit the source file**: `waveorder_quickstart.py` (Jupytext format)

2. **Generate the notebook**:
   ```bash
   jupytext --to ipynb waveorder_quickstart.py
   ```

3. **Test the notebook** end-to-end:
   ```bash
   jupyter notebook waveorder_quickstart.ipynb
   ```

   The notebook will automatically download demo data on first run.

4. **Commit only the source**:
   ```bash
   git add waveorder_quickstart.py README.md .gitignore
   git commit -m "Update WaveOrder quickstart notebook"
   ```

#### What Gets Committed vs. Downloaded

| File | Size | Commit? | Why? |
|------|------|---------|------|
| `waveorder_quickstart.py` | ~27 KB | ✅ Yes | Source code |
| `README.md` | Small | ✅ Yes | Documentation |
| `.gitignore` | Small | ✅ Yes | Git configuration |
| `waveorder/20x.zarr` | ~123 MB | ❌ No | Downloaded from public server |
| `waveorder_quickstart.ipynb` | Varies | ❌ No | Generated from .py |

### Demo Data

The demo data is hosted on CZ Biohub's public server:
- **URL**: https://public.czbiohub.org/comp.micro/neurips_demos/waveorder/20x.zarr
- **Size**: ~123 MB
- **Format**: OME-Zarr
- **Contents**: 3D label-free microscopy z-stack

The notebook automatically downloads this file using `wget` if not already present.

## Files

## Publishing Checklist

Before publishing:

- [ ] Notebook runs end-to-end in Colab
- [ ] All cells execute without errors
- [ ] Visualizations render correctly
- [ ] Optimization converges
- [ ] Final comparison shows improvement
- [ ] Embedded data extracts correctly
- [ ] File size is reasonable (<10 MB)

## References

- Chandler T., Ivanov I.E., Hirata-Miyasaki E., et al. "WaveOrder: Physics-informed ML for auto-tuned multi-contrast computational microscopy from cells to organisms." [arXiv:2412.09775](https://arxiv.org/abs/2412.09775) (2025).

- WaveOrder GitHub: https://github.com/mehta-lab/waveorder
- WaveOrder Documentation: https://mehta-lab.github.io/waveorder/
- WaveOrder PyPI: https://pypi.org/project/waveorder/

## Contact

For questions or issues:
- **GitHub Issues**: https://github.com/mehta-lab/waveorder/issues
- **Documentation**: https://mehta-lab.github.io/waveorder/
- **Model Card**: https://virtualcellmodels.cziscience.com/model/waveorder

## License

BSD-3-Clause (matches WaveOrder software license)
