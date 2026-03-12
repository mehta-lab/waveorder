# Plan: Projection Modeling Simulation Script

## Goal
Create simulation scripts that simulate images as follows, and then implement the reconstruction algorithm.

Forward simulation
1. Generate three distinct 3D test phantoms (isolated bead, line `[o]` pattern, Shepp-Logan)
2. Convert each to fluorescence density and phase density
3. Simulate fluorescence and phase images via transfer functions
4. Save all volumes to an OME-Zarr v3 store (ome-ngff 0.5) via iohub
5. Compute Siddon sum-projections of simulated images at -70 to +70 degrees in 5-degree steps
6. Assume the black level of 100 and the peak photon flux of 1024, scale the simulated images (but not the test phantom) that sets the value of brightest voxel to 1024. Add Poisson noise to mimic realistic imaging.

Inverse algorithm - geometric:
1. From the noisy projections of original object at angles +- x degrees, estimate the projection along the 0 degree. Ignore the blur and map the phantoms to intensities. In other words, simulate the existence of the solution.

Inverse algorithm - wave optical:
1. Repeat the above process for images of 3 targets acquired with with blur, and estimate the projection along z-axis from projections made at angles +-x degrees. In other words, use 6 sets of projection images to estimate the object.


Metrics:
Compute the metrics of similarity between the original object and reconstruction. Make a note of these metrics for all targets, their images, and inverse algorithm. Choose standard Tikhonov regularized inverse algorithms.


## Output
Forward simulation CLI: `docs/examples/demos/projection_modeling/projection_modeling.py`
Inverse algorithm - geometric: `docs/examples/demos/projection_modeling/projection_reconstruction_geometric.py`
Inverse algorithm - wave: `docs/examples/demos/projection_modeling/projection_reconstruction_wave.py`


---

## CLI Usage

The forward simulation is a Click CLI with three subcommands that run in sequence.
Each subcommand reads the output of the previous stage from the shared store.

```bash
cd docs/examples/demos/projection_modeling/

# Step 1: Generate 3D test phantoms → object column
uv run python projection_modeling.py object --data-dir ./data

# Step 2: Blur + noise → rawimage column
uv run python projection_modeling.py image --data-dir ./data

# Step 3: Siddon projections → projections column
uv run python projection_modeling.py project --data-dir ./data

# Show help
uv run python projection_modeling.py -h
uv run python projection_modeling.py object -h
```

---

## Data Format: OME-Zarr v3 via iohub

All scripts write to a single HCS-layout OME-Zarr store using `iohub.ngff.open_ome_zarr` with `version="0.5"`. This produces zarr v3 stores with ome-ngff 0.5 metadata, viewable in napari via `napari-ome-zarr`.

### Store Layout

```
projection_modeling.zarr/          # HCS plate, ome-ngff 0.5
  point/                           # row = sample_type
    object/                        # col: ground-truth densities
      0/                           # FOV: TCZYX (1, 2, 256, 256, 256)
    rawimage/                      # col: blurred + noisy images
      0/                           # FOV: TCZYX (1, 2, 256, 256, 256)
    projections/                   # col: Siddon projection stacks
      0/                           # FOV: TCZYX (1, 2, 29, 256, X_pad)
    recongeo/                      # col: geometric reconstruction
      0/
    reconwave/                     # col: wave-optical reconstruction
      0/
  lines/
    (same columns)
  shepplogan/
    (same columns)
```

**Note:** HCS plate axis names must be alphanumeric. Use `rawimage`, `recongeo`, `reconwave`, `shepplogan` (no underscores or hyphens).

Each subcommand creates its own column positions when it runs:
- `object` creates the store fresh and writes `object` positions
- `image` opens the store in `r+` mode and creates `rawimage` positions
- `project` opens the store in `r+` mode and creates `projections` positions

### Channel Naming

All columns share channel names `["Fluorescence", "Phase"]` (C=0 and C=1).

| Column | Shape | Description |
|--------|-------|-------------|
| `object` | `(1, 2, 256, 256, 256)` | Ground-truth densities, normalized [0, 1] |
| `rawimage` | `(1, 2, 256, 256, 256)` | Blurred, scaled to [100, 1024], Poisson noise |
| `projections` | `(1, 2, 29, 256, X_pad)` | Siddon sum-projections at -70:5:70 deg |
| `recongeo` | `(1, 2, 256, 256, 256)` | Geometric reconstruction (no OTF) |
| `reconwave` | `(1, 2, 256, 256, 256)` | Wave-optical reconstruction (with OTF) |

### Projection Stack Convention

Projections are stored as 3D stacks browsable with a standard Z-slider.
Each slice along the Z-axis is one projection angle.
The Z-scale in the transform metadata encodes the angular step: `scale[2] = 5.0` (degrees).
Actual angles are stored in position zattrs:

```python
pos.zattrs["projection_angles_deg"]  # [-70, -65, ..., 65, 70]
pos.zattrs["angle_step_deg"]         # 5
pos.zattrs["angle_start_deg"]        # -70
```

Since Siddon projections at oblique angles produce wider lateral output,
all projections are center-padded to the maximum width (363 pixels for a 256^3 cube at 70 deg).

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

Module-level constants at the top of the CLI script:

```python
BEAD_RADIUS = 0.25              # um (0.5 um diameter)
BEAD_INDEX = 1.52               # refractive index of beads/structures
MEDIA_INDEX = 1.33              # water
LINE_RADIUS = 0.25              # um (0.5 um line thickness → radius)
BRACKET_HALF_EXTENT = 2.0       # um, half-height of brackets in Y
BRACKET_X_OFFSET = 2.0          # um, bracket distance from center in X
BRACKET_CAP_LENGTH = 0.6        # um, horizontal cap length for [ and ]
TORUS_MAJOR_RADIUS = 0.8        # um, ring radius for 'o'

VOLUME_EXTENT = 12.8            # um (256 voxels at 50 nm)
VOXEL_SIZE = 0.05               # um (50 nm isotropic sampling)

WAVELENGTH_ILLUMINATION = 0.500  # um
WAVELENGTH_EMISSION = 0.520      # um (Stokes shift)
NA_DETECTION = 1.0
NA_ILLUMINATION = 0.5
Z_PADDING = 0
BLACK_LEVEL = 100                # detector black level (counts)
PEAK_INTENSITY = 1024            # peak photon flux (counts)

PROJECTION_ANGLES = list(range(-70, 75, 5))  # 29 angles
```

Derived shape: `ZYX_SHAPE = (256, 256, 256)`.

## Part 2: Three Test Phantoms

### 2a: Isolated bead

A single sphere at the volume center with radius 0.25 um (0.5 um diameter). Binary volume: voxels within `bead_radius` of center set to 1.

### 2b: Line `[o]` pattern

Bracket-shaped cylinders and a torus ring, all with 0.5 um tube diameter:

- **`[`**: Three segments — vertical bar at `x = -bracket_x_offset`, plus horizontal caps at `y = ±bracket_half_extent` extending rightward by `bracket_cap_length`.
- **`]`**: Mirror of `[` — vertical bar at `x = +bracket_x_offset`, caps extending leftward.
- **`o`**: Torus of radius `torus_major_radius` centered at the origin.

Gap between brackets and ring: `bracket_x_offset - torus_major_radius - line_radius = 0.95 um`.

### 2c: Shepp-Logan

Standard 10-ellipsoid 3D Shepp-Logan phantom. Density normalized to [0, 1].

## Part 3: Conversion to Fluorescence and Phase Density

Each phantom volume (values 0–1) is converted to fluorescence density (direct mapping) and phase (delta_n * voxel_size / wavelength_medium). Uniform transmission assumed.

## Part 4: Forward Simulation (image subcommand)

Blur fluorescence density through the fluorescence OTF, phase density through the phase transfer function. Scale each to [BLACK_LEVEL, PEAK_INTENSITY] and apply Poisson noise.

## Part 5: Projections (project subcommand)

For each sample, read the rawimage volumes, compute Siddon sum-projections at 29 angles (-70 to +70, step 5). Center-pad oblique projections to uniform width. Stack into a TCZYX array where Z indexes angle.

## Part 6: Reconstruction (separate scripts)

### Geometric reconstruction (projection_reconstruction_geometric.py)
Read `object` stage, compute Siddon projections at ±x degrees, run CG-Tikhonov, write to `recongeo`.

### Wave-optical reconstruction (projection_reconstruction_wave.py)
Read `rawimage` stage, write to `reconwave`.

## Testing

```bash
cd docs/examples/demos/projection_modeling/

# Run all forward simulation stages
uv run python projection_modeling.py object
uv run python projection_modeling.py image
uv run python projection_modeling.py project

# Verify store structure
uv run python -c "
from iohub.ngff import open_ome_zarr
with open_ome_zarr('data/projection_modeling.zarr', mode='r') as p:
    for name, pos in p.positions():
        print(f'{name}: shape={pos[\"0\"].shape}, channels={pos.channel_names}')
"
```
