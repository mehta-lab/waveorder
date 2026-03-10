# Plan: Projection Modeling Simulation Script

## Goal
Create a simulation script that:
1. Generates three distinct 3D test phantoms (isolated bead, line `[o]` pattern, Shepp-Logan)
2. Converts each to fluorescence density and phase density
3. Simulates fluorescence and phase images via transfer functions
4. Saves all volumes to a zarr store for inspection
5. Computes average and max projections of the original and blurred datasets with +- 60 degree angle. Use Siddon algorithm where appropriate.

## Output
A single script: `docs/examples/demos/projection_modeling/projection_modeling.py`

---

## Part 1: Configurable Parameters

Define all simulation parameters as variables at the top of the script:

```python
# Object parameters
bead_radius = 0.25          # um (0.5 um diameter)
bead_index = 1.52           # refractive index of beads/structures
media_index = 1.33          # water
line_radius = 0.25          # um (0.5 um line thickness → radius)
pattern_extent = 2.5        # um, half-extent of [o] pattern around center
line_spacing = 2.0          # um, spacing for bracket lines

# Volume parameters
volume_extent_x = 10.0      # um
volume_extent_y = 10.0      # um
volume_extent_z = 10.0      # um
voxel_size = 0.05           # um (50 nm isotropic sampling)

# Imaging parameters
wavelength_illumination = 0.500  # um
wavelength_emission = 0.520      # um (Stokes shift)
na_detection = 1.0
na_illumination = 0.5
z_padding = 0
```

Derived shape: `zyx_shape = (200, 200, 200)` from `volume_extent / voxel_size`.

## Part 2: Three Test Phantoms

### 2a: Isolated bead

A single sphere at the volume center with radius 0.25 um (0.5 um diameter). Binary volume: voxels within `bead_radius` of center set to 1.

```python
def generate_isolated_bead(zyx_shape, voxel_size, bead_radius):
    """Single sphere at volume center. Returns binary volume (0/1)."""
```

### 2b: Line `[o]` pattern

Extended thin structures using lines (cylinders) and a ring (torus), all with 0.5 um tube diameter:

- **`[` and `]`**: Vertical line segments (cylinders along Y) at `x = ±pattern_extent/2`, spanning `y ∈ [-pattern_extent/2, +pattern_extent/2]`, in the central XY plane (`z = 0`).
- **`o`**: A torus (ring) of radius ~0.8 um centered at the origin, in the central XY plane, tube radius = `line_radius`.

For each voxel, compute the distance to the nearest centerline. Voxels within `line_radius` are set to 1. Apply a small Gaussian blur for smooth edges.

```python
def generate_line_pattern(zyx_shape, voxel_size, line_radius, pattern_extent, line_spacing):
    """[o] pattern using lines (cylinders) and a ring (torus).

    For each voxel, compute distance to nearest centerline:
    - '[' and ']': vertical line segments at x = ±pattern_extent/2,
      y ∈ [-pattern_extent/2, +pattern_extent/2], z = 0
    - 'o': circle of radius ~0.8 um in central XY plane

    Voxels within line_radius of any centerline are set to 1.
    Apply small Gaussian blur for smooth edges.
    """
```

### 2c: Shepp-Logan

Standard 10-ellipsoid 3D Shepp-Logan phantom, unchanged from the existing implementation. Density normalized to [0, 1].

```python
def generate_shepp_logan_3d(zyx_shape):
    """3D Shepp-Logan phantom. Returns density array normalized to [0, 1]."""
```

## Part 3: Conversion to Fluorescence and Phase Density

Each phantom volume (values 0–1) is converted to two physical quantities:

```python
def phantom_to_fluorescence_and_phase(volume, voxel_size, bead_index, media_index, wavelength_illumination):
    fluorescence = volume  # density 0-1
    wavelength_medium = wavelength_illumination / media_index
    delta_n = (bead_index - media_index) * volume
    phase = delta_n * voxel_size / wavelength_medium  # cycles/voxel
    return fluorescence, phase
```

No absorption channel.

## Part 4: Forward Simulation

For each phantom, blur fluorescence density through the fluorescence OTF and phase density through the phase transfer function.

### Fluorescence simulation
```python
otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
    zyx_shape, voxel_size, voxel_size, wavelength_emission,
    z_padding, media_index, na_detection
)
fluorescence_blurred = isotropic_fluorescent_thick_3d.apply_transfer_function(
    fluorescence_density, otf, z_padding
)
```

### Phase simulation
```python
real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
    zyx_shape, voxel_size, voxel_size, wavelength_illumination,
    z_padding, media_index, na_illumination, na_detection
)
phase_blurred = phase_thick_3d.apply_transfer_function(
    phase_density, real_tf, z_padding, brightness=1e3
)
```

## Part 5: Zarr Store Structure

```
projection_modeling.zarr/
  isolated_bead/
    fluorescence_density, phase_density, fluorescence_blurred, phase_blurred
  line_pattern/
    fluorescence_density, phase_density, fluorescence_blurred, phase_blurred
  shepp_logan/
    fluorescence_density, phase_density, fluorescence_blurred, phase_blurred
```

Metadata stored as root attributes.

## Part 6: Projections via Siddon's Algorithm

Compute average and max projections of each volume (original density and blurred) at -60°, 0°, and +60°.

- **0° projections**: simple sum/max along Z (exact, no ray tracing needed).
- **±60° projections**: Siddon's ray-tracing algorithm traces each ray through the voxel grid in the ZX plane. Since rays have no Y component, each lateral detector position shares the same (Z, X) voxel sequence across all Y rows, enabling vectorized gathering over Y.
- **Average projection**: Siddon sum projection divided by the total ray path length through the volume.

Each phantom group stores 24 projection arrays:
4 volume types × 3 angles × 2 modes (avg, max).

## Part 7: Visualization

Use `view_volumes.py` (neuroglancer) to inspect the zarr store. The viewer iterates over all groups dynamically and auto-adjusts contrast range per layer.

## Implementation Order

1. Write configurable parameters section
2. Implement `generate_isolated_bead()`
3. Implement `generate_line_pattern()`
4. Keep `generate_shepp_logan_3d()` unchanged
5. Implement `phantom_to_fluorescence_and_phase()`
6. Implement `siddon_project()`
7. Compute fluorescence and phase transfer functions
8. Loop over three phantoms: convert, blur, save to zarr
9. Loop over three phantoms: project all volumes at all angles, save to zarr
10. Verify: 3 groups × (4 volumes + 24 projections) = 84 arrays
