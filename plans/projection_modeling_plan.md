# Plan: Projection Modeling Simulation Script
## Simulations
* Objects: 1D - point, 2D - Biohub logo [o], 3D - Shepp Logan
* Models and reconstructions: limited angle tomography over $\pm$ 70 degrees, two view limited angle tomography, limited angle tomography with optical blur, and two view limited angle tomography with optical blur.
 
## Scripts

| Script | Purpose |
|--------|---------|
| `projection_modeling.py` | Forward simulation CLI: `object`, `image`, `project` |
| `projection_reconstruction.py` | Reconstruction CLI: `geometric`, `wave`, `geometric-two-projections`, `wave-two-projections` |
| `geometric_two_projections_sweep.py` | Two-view sweep: PSNR vs theta from 5 to 70 deg |
| `visualize_reconstruction.py` | Plotting: forward, geometric, wave figures |
| `siddon.py` | `SiddonOperator` (sparse matmul), `cg_tikhonov`, `ramp_filter_sinogram` |

All scripts live in `docs/examples/demos/projection_modeling/`.

## Store Layout

`data/projection_modeling.zarr` — HCS plate (ome-ngff 0.5, iohub `version="0.5"`).
Rows: `point`, `lines`, `shepplogan`.
Columns: `object`, `rawimage`, `projections`, `recongeo`, `reconwave`, `recongeo2`, `reconwave2`.
Shape: TCZYX `(1, 2, 256, 256, 256)`, channels `["Fluorescence", "Phase"]`.

## Forward Models

**Geometric:** `H(x) = [siddon_project(x, theta) for theta in angles]`. No blur.
**Wave-optical:** `H(x) = [siddon_project(OTF * x, theta) for theta in angles]`. 3D OTF via FFT.

Both solved via CG-Tikhonov: `(H*H + lambda I)x = H*y`.

## CLI Quick Reference

```bash
cd docs/examples/demos/projection_modeling/

# Forward simulation
uv run python projection_modeling.py object   --data-dir ./data
uv run python projection_modeling.py image    --data-dir ./data
uv run python projection_modeling.py project  --data-dir ./data

# Reconstruction
uv run python projection_reconstruction.py geometric --data-dir ./data
uv run python projection_reconstruction.py wave      --data-dir ./data
uv run python projection_reconstruction.py geometric-two-projections --angle 30 --data-dir ./data
uv run python projection_reconstruction.py wave-two-projections      --angle 30 --data-dir ./data

# Two-view sweep
uv run python geometric_two_projections_sweep.py --data-dir ./data --out-dir ./plots

# Visualize
uv run python visualize_reconstruction.py plot --sample all
```
