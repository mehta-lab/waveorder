# Plan: Projection Modeling Simulation Script

## Goal
Create simulation scripts that simulate images as follows, and then implement the reconstruction algorithm.

Forward simulation
1. Generate three distinct 3D test phantoms (isolated bead, line `[o]` pattern, Shepp-Logan)
2. Convert each to fluorescence density and phase density
3. Simulate fluorescence and phase images via transfer functions
4. Save all volumes to an OME-Zarr v3 store (ome-ngff 0.5) via iohub
5. Compute Siddon mean-projections of simulated images at -70 to +70 degrees in 5-degree steps
6. Assume the black level of 100 and the peak photon flux of 1024, scale the simulated images (but not the test phantom) that sets the value of brightest voxel to 1024. Add Poisson noise to mimic realistic imaging.

Inverse algorithm — four subcommands:
1. **geometric** (limited-angle tomography without blur): Reconstruct from ALL projections of the unblurred object using Siddon-only forward model. Tests whether the CG-Tikhonov solver recovers the object from clean geometric projections.
2. **wave** (limited-angle tomography with blur): Reconstruct from ALL projections of the blurred+noisy rawimage using 3D OTF + Siddon forward model. Tests simultaneous deconvolution and reconstruction.
3. **geometric-limited** (limited-angle tomography with two views, no blur): Reconstruct from a single ±theta projection pair of the unblurred object. No blur in forward model.
4. **wave-limited** (limited-angle tomography with two views and blur): Reconstruct from a single ±theta projection pair of the blurred+noisy rawimage. 3D OTF in forward model.

Fourier-slice theorem: projecting an OTF-blurred 3D volume at angle theta is equivalent to applying the central slice of the 3D OTF at that angle as a 2D transfer function. Using the full 3D OTF convolution in the forward model naturally accounts for angle-dependent resolution and defocus coupling.

Metrics:
MSE and PSNR between reconstruction and unblurred ground truth (`object` column). Stored in position zattrs.


## Output
Forward simulation CLI: `docs/examples/demos/projection_modeling/projection_modeling.py`
Reconstruction CLI: `docs/examples/demos/projection_modeling/projection_reconstruction.py`

Legacy scripts (superseded):
- `projection_reconstruction_geometric.py`
- `projection_reconstruction_wave.py`


---

## CLI Usage

```bash
cd docs/examples/demos/projection_modeling/

# === Forward simulation (three stages) ===
uv run python projection_modeling.py object   --data-dir ./data
uv run python projection_modeling.py image    --data-dir ./data
uv run python projection_modeling.py project  --data-dir ./data

# === Reconstruction (four subcommands, extend existing store) ===
# All 29 projection angles
uv run python projection_reconstruction.py geometric --data-dir ./data
uv run python projection_reconstruction.py wave      --data-dir ./data

# Single +/-theta pair
uv run python projection_reconstruction.py geometric-limited --angle 30 --data-dir ./data
uv run python projection_reconstruction.py wave-limited      --angle 30 --data-dir ./data

# Tune reconstruction parameters
uv run python projection_reconstruction.py geometric --reg 1e-4 --niter 100
uv run python projection_reconstruction.py wave --reg 1e-2 --niter 30

# Show help
uv run python projection_reconstruction.py -h
uv run python projection_reconstruction.py wave -h
```

---

## Data Format: OME-Zarr v3 via iohub

All scripts write to a single HCS-layout OME-Zarr store using `iohub.ngff.open_ome_zarr` with `version="0.5"`.

### Store Layout

```
projection_modeling.zarr/          # HCS plate, ome-ngff 0.5
  point/                           # row = sample_type
    object/                        # ground-truth densities  (1, 2, 256, 256, 256)
      0/
    rawimage/                      # blurred + noisy images  (1, 2, 256, 256, 256)
      0/
    projections/                   # Siddon projection stacks (1, 2, 29, 256, 363)
      0/
    recongeo/                      # limited-angle tomography without blur
      0/
    reconwave/                     # limited-angle tomography with blur
      0/
    recongeoL/                     # two-view tomography without blur
      0/
    reconwaveL/                    # two-view tomography with blur
      0/
  lines/
    (same columns)
  shepplogan/
    (same columns)
```

**Note:** HCS axis names must be alphanumeric. Column names: `rawimage`, `recongeo`, `reconwave`, `recongeoL`, `reconwaveL`, `shepplogan`.

Each script/subcommand creates its own column positions when it runs. Positions that already exist are reused (data overwritten).

### Channel Naming

All columns share channel names `["Fluorescence", "Phase"]` (C=0 and C=1).

| Column | Shape | Source | Forward model |
|--------|-------|--------|---------------|
| `object` | `(1, 2, 256, 256, 256)` | Phantom generator | — |
| `rawimage` | `(1, 2, 256, 256, 256)` | OTF blur + noise | — |
| `projections` | `(1, 2, 29, 256, 363)` | Siddon mean-proj of rawimage | — |
| `recongeo` | `(1, 2, 256, 256, 256)` | object → Siddon | Limited-angle tomography, no blur |
| `reconwave` | `(1, 2, 256, 256, 256)` | rawimage → OTF+Siddon | Limited-angle tomography, with blur |
| `recongeoL` | `(1, 2, 256, 256, 256)` | object → Siddon | Two-view tomography, no blur |
| `reconwaveL` | `(1, 2, 256, 256, 256)` | rawimage → OTF+Siddon | Two-view tomography, with blur |

### Reconstruction Metadata

Each reconstruction position stores metrics and parameters in zattrs:

```python
pos.zattrs["Fluorescence_mse"]   # MSE vs ground truth
pos.zattrs["Fluorescence_psnr"]  # PSNR in dB
pos.zattrs["Phase_mse"]
pos.zattrs["Phase_psnr"]
pos.zattrs["angles"]             # list of projection angles used
pos.zattrs["reg_strength"]       # Tikhonov lambda
pos.zattrs["n_iter"]             # CG iterations
```

### Projection Stack Convention

Projections are stored as 3D stacks browsable with a standard Z-slider.
Each Z-slice is one projection angle. Z-scale = 5.0 (angular step in degrees).
Actual angles stored in `pos.zattrs["projection_angles_deg"]`.
Oblique projections center-padded to maximum width (363 px for 256^3 cube at 70 deg).

### Viewing

```bash
# Programmatic (works with zarr v3):
python -c "
import napari; from iohub.ngff import open_ome_zarr; import numpy as np
plate = open_ome_zarr('data/projection_modeling.zarr', mode='r')
viewer = napari.Viewer()
for name, pos in plate.positions():
    data = np.array(pos['0'])
    for c, ch in enumerate(pos.channel_names):
        viewer.add_image(data[0, c], name=f'{name}/{ch}', visible=False)
napari.run(); plate.close()
"
```

---

## Part 1: Configurable Parameters

Module-level constants at the top of each CLI script:

```python
VOXEL_SIZE = 0.05               # um (50 nm isotropic)
ZYX_SHAPE = (256, 256, 256)     # 12.8 um cube

WAVELENGTH_ILLUMINATION = 0.500  # um
WAVELENGTH_EMISSION = 0.520      # um
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
MEDIA_INDEX = 1.33
BLACK_LEVEL = 100                # detector offset (counts)
PEAK_INTENSITY = 1024            # peak signal (counts)

PROJECTION_ANGLES = list(range(-70, 75, 5))  # 29 angles
REG_STRENGTH = 1e-3              # default Tikhonov lambda
N_ITER = 50                      # default CG iterations
```

## Part 2: Three Test Phantoms

### 2a: Isolated bead
Single sphere at volume center, radius 0.25 um. Binary volume.

### 2b: Line `[o]` pattern
Bracket-shaped cylinders and a torus ring, 0.5 um tube diameter.
Gap between brackets and ring: 0.95 um (no overlap).

### 2c: Shepp-Logan
Standard 10-ellipsoid 3D phantom. Density normalized to [0, 1].

## Part 3: Forward Simulation

`projection_modeling.py image` blurs each channel through the 3D transfer function:
- Fluorescence: `isotropic_fluorescent_thick_3d` OTF
- Phase: `phase_thick_3d` real transfer function

Then scales to [BLACK_LEVEL, PEAK_INTENSITY] and applies Poisson noise.

## Part 4: Projections

`projection_modeling.py project` computes Siddon mean-projections (sum / physical path length) at 29 angles. All projections padded to uniform width and stacked along Z.

## Part 5: Reconstruction

`projection_reconstruction.py` implements four subcommands sharing one core CG-Tikhonov solver.

### Forward models

**Geometric**: `forward(x) = [siddon_project(x, angle) for angle in angles]`

**Wave-optical**: `forward(x) = [siddon_project(OTF_3D ⊛ x, angle) for angle in angles]`

The wave adjoint is: `adjoint(y) = OTF_3D^† ⊛ sum(siddon_backproject(y_i, angle_i))`

OTF convolution uses PyTorch FFT on GPU (`torch.fft.fftn`/`ifftn`). Siddon ray-tracing and CG iterations remain on CPU (numpy).

### Angle configurations

- **All angles** (`geometric`, `wave`): Limited-angle tomography with 29 projections at -70:5:70 degrees
- **Two views** (`geometric-limited`, `wave-limited`): Limited-angle tomography with a single ±theta pair (2 projections)

### Fourier-slice theorem

Projecting a 3D OTF-blurred volume at angle theta samples the central slice of the 3D Fourier transform at that angle. The 3D OTF convolution in the forward model correctly weights each Fourier slice by the angle-dependent microscope transfer function, coupling defocus and projection geometry. This is why the wave-optical forward model applies the full 3D OTF (not an angle-dependent 2D OTF) before Siddon projection.

## Testing

```bash
cd docs/examples/demos/projection_modeling/

# Run full pipeline
uv run python projection_modeling.py object
uv run python projection_modeling.py image
uv run python projection_modeling.py project
uv run python projection_reconstruction.py geometric
uv run python projection_reconstruction.py wave

# Limited reconstructions
uv run python projection_reconstruction.py geometric-limited --angle 30
uv run python projection_reconstruction.py wave-limited --angle 30

# Verify store
uv run python -c "
from iohub.ngff import open_ome_zarr
with open_ome_zarr('data/projection_modeling.zarr', mode='r') as p:
    for name, pos in p.positions():
        print(f'{name}: shape={pos[\"0\"].shape}, channels={pos.channel_names}')
"
```
