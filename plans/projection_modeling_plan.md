# Plan: Projection Modeling Simulation Script

## Goal
Create a simulation script that:
1. Generates a custom 3D phantom with beads arranged in a `[o]` pattern
2. Generates a miniaturized Shepp-Logan phantom modeled as an absorbing specimen
3. Simulates fluorescence and phase images of both objects
4. Computes mean and max projections at different angles relative to the z-axis
5. Visualizes all results

## Output
A single script: `docs/examples/demos/projection_modeling/projection_modeling.py`

---

## Part 1: Configurable Parameters

Define all simulation parameters as variables at the top of the script:

```python
# Object parameters
bead_radius = 0.25          # um (0.5 um diameter)
bead_index = 1.52           # refractive index of beads
media_index = 1.33          # water
bead_spacing = 1.0          # um, center-to-center in [o] pattern
pattern_extent = 2.5        # um, extent of [o] pattern around center

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

# Projection parameters
projection_angles = [0, 15, 30, 45, 60, 75, 90]  # degrees from z-axis
```

Derived shape: `zyx_shape = (200, 200, 200)` from `volume_extent / voxel_size`.

## Part 2: `[o]` Bead Pattern Phantom

The pattern `[o]` is composed of bead positions in the central XY plane (z = center):
- **`[`**: A vertical line of beads on the left, spaced by `bead_spacing`
- **`o`**: A ring/circle of beads in the center
- **`]`**: A vertical line of beads on the right, spaced by `bead_spacing`

The pattern occupies 2.5 um around the center of the volume.

**Implementation approach:**
1. Compute bead center coordinates (in um) relative to volume center.
2. For `[` and `]`: vertical columns of beads at x = -pattern_extent/2 and x = +pattern_extent/2, spanning y from -pattern_extent/2 to +pattern_extent/2 with `bead_spacing` steps.
3. For `o`: a circle of beads at center, radius ~0.8 um, with beads placed every `bead_spacing` arc-length around the circle.
4. All beads placed at z = 0 (central plane).
5. For each bead center, paint a smooth sphere (Gaussian-blurred) into the 3D volume.
6. Produce two arrays:
   - `zyx_fluorescence_density`: binary/smooth bead locations (values 0–1)
   - `zyx_phase`: convert density to phase in cycles/voxel using `(n_bead - n_media) * z_pixel_size / wavelength_media`

## Part 3: Shepp-Logan Phantom

Implement a miniaturized 3D Shepp-Logan phantom:
1. Use the standard Shepp-Logan ellipsoid parameters (10 ellipsoids with defined centers, semi-axes, rotation angles, and densities).
2. Scale to fit within the 10 um volume.
3. Model as an **absorbing** specimen:
   - The Shepp-Logan density values represent absorption coefficient.
   - For phase imaging: the imaginary potential transfer function captures absorption.
   - Use `absorption_ratio` parameter in `phase_thick_3d.apply_inverse_transfer_function()` for reconstruction.
4. Also generate a fluorescence density version (absolute value of density) for fluorescence forward simulation.

**Implementation:** Write a `generate_shepp_logan_3d(zyx_shape, voxel_size)` function that returns a 3D tensor by iterating over the standard ellipsoid definitions and testing which voxels fall inside each ellipsoid (with rotation).

## Part 4: Forward Simulation

For **each** phantom (bead pattern and Shepp-Logan):

### Fluorescence simulation
Following `isotropic_fluorescent_thick_3d` pattern:
```python
otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
    zyx_shape, voxel_size, voxel_size, wavelength_emission,
    z_padding, media_index, na_detection
)
zyx_fluorescence_data = isotropic_fluorescent_thick_3d.apply_transfer_function(
    zyx_fluorescence_density, otf, z_padding
)
```

### Phase simulation
Following `phase_thick_3d` pattern:
```python
real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
    zyx_shape, voxel_size, voxel_size, wavelength_illumination,
    z_padding, media_index, na_illumination, na_detection
)
zyx_phase_data = phase_thick_3d.apply_transfer_function(
    zyx_phase, real_tf, z_padding, brightness=1e3
)
```

For the Shepp-Logan (absorbing), both `real_tf` and `imag_tf` are available from `phase_thick_3d.calculate_transfer_function()`. The imaginary component captures absorption contrast. We can simulate absorption by treating the Shepp-Logan densities as the imaginary part of the potential and using the same forward model.

## Part 5: Projections at Angles via Siddon's Algorithm

Use Siddon's ray-tracing algorithm to compute projections. Siddon's algorithm traces each ray through the voxel grid and accumulates exact intersection lengths per voxel — producing accurate line integrals without rotating the volume.

### Why Siddon over rotate-then-project

- **Accuracy:** Siddon computes exact ray–voxel intersection lengths; rotate-then-project introduces interpolation artifacts and loses voxels at the volume boundary.
- **Physicality:** Line integrals along rays match the actual projection geometry (Beer–Lambert for absorption, fluorescence integration along the optical path).
- **Max projection:** Siddon naturally supports max-along-ray by tracking the maximum voxel value encountered, avoiding interpolation bias from rotating the volume.

### Algorithm outline (Siddon 1985)

For a ray from source to detector pixel, parameterize as `P(t) = source + t * direction`:
1. Compute the parametric intersections `t_x`, `t_y`, `t_z` with every grid plane along each axis.
2. Merge and sort all intersection parameters.
3. Between consecutive `t` values, the ray traverses one voxel; the intersection length is `(t_{k+1} - t_k) * ||direction||`.
4. Accumulate: sum projection = `Σ (length_k * voxel_value_k)`; max projection = `max(voxel_value_k)`.

### Projection geometry

Rays propagate in the ZX plane, tilted by `angle` from the Z-axis. For a given angle θ:
- Ray direction: `d = (sin θ, 0, cos θ)` (rotation around Y-axis).
- Detector plane: perpendicular to `d`, centered on the volume.
- Each detector pixel maps to one ray through the volume.

### Implementation

```python
def siddon_project(volume, angle_deg, voxel_size, mode="sum"):
    """Project a 3D volume at a given angle using Siddon's algorithm.

    Parameters
    ----------
    volume : ndarray, shape (Z, Y, X)
    angle_deg : float
        Tilt angle from the Z-axis, rotation around Y.
    voxel_size : float
        Isotropic voxel spacing in um.
    mode : str
        "sum" for line-integral (sum) projection,
        "max" for max-intensity projection.

    Returns
    -------
    projection : ndarray, shape (Y, X')
        2D projection image. X' may differ from X for large angles.
    """
    theta = np.radians(angle_deg)
    nz, ny, nx = volume.shape
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])

    # Detector grid: Y unchanged, lateral axis spans the
    # bounding box of the rotated volume to avoid clipping.
    lateral_extent = nz * abs(np.sin(theta)) + nx * abs(np.cos(theta))
    n_lateral = int(np.ceil(lateral_extent))
    projection = np.zeros((ny, n_lateral))

    for iy in range(ny):
        for il in range(n_lateral):
            # Map detector pixel (il, iy) to ray origin
            # (perpendicular offset from center along detector lateral axis)
            offset = (il - n_lateral / 2) * voxel_size
            origin = compute_ray_origin(offset, iy, direction, volume.shape, voxel_size)

            # Trace ray through grid
            voxel_indices, lengths = siddon_trace(origin, direction, volume.shape, voxel_size)

            if mode == "sum":
                projection[iy, il] = sum(
                    lengths[k] * volume[iz, iy, ix] for k, (iz, _, ix) in enumerate(voxel_indices)
                )
            elif mode == "max":
                projection[iy, il] = max(
                    (volume[iz, iy, ix] for iz, _, ix in voxel_indices), default=0.0
                )
    return projection
```

The inner `siddon_trace` function implements the parametric intersection logic from Siddon (1985). For performance, vectorize the per-ray loop with NumPy or use Numba `@jit`.

### Wrapper for multiple angles

```python
def compute_projections(volume, angles, voxel_size, modes=("sum", "max")):
    """Compute projections at multiple angles.

    Returns
    -------
    dict : {angle: {"sum": 2d_array, "max": 2d_array}}
    """
    results = {}
    for angle in angles:
        results[angle] = {}
        for mode in modes:
            results[angle][mode] = siddon_project(volume, angle, voxel_size, mode=mode)
    return results
```

### Projections before and after microscope blur

Compute projections at two stages to isolate the effect of microscope blur:

1. **Pre-blur projections** — project the ground-truth phantom directly:
   ```python
   proj_ideal = compute_projections(phantom_volume, projection_angles, voxel_size)
   ```
2. **Post-blur projections** — project the microscope-blurred volume:
   ```python
   blurred_volume = apply_transfer_function(phantom_volume, otf, z_padding)
   proj_blurred = compute_projections(blurred_volume, projection_angles, voxel_size)
   ```

This separation lets us compare ideal geometric projections against what the microscope actually delivers, and quantify resolution loss from the OTF at each projection angle.

## Part 6: Visualization

Use `matplotlib` (not napari) for non-interactive visualization that works in scripts and notebooks:

1. **Figure 1 — Bead phantom overview:**
   - Row 1: Central XY slice of phantom, fluorescence simulation, phase simulation
   - Row 2: XZ cross-section of each

2. **Figure 2 — Shepp-Logan phantom overview:**
   - Same layout as Figure 1

3. **Figure 3 — Pre-blur projections (bead phantom fluorescence density):**
   - Row 1: Sum projections at each angle (one subplot per angle)
   - Row 2: Max projections at each angle

4. **Figure 4 — Post-blur projections (bead phantom fluorescence):**
   - Same grid layout as Figure 3, but from the microscope-blurred volume

5. **Figure 5 — Pre-blur vs post-blur comparison (bead phantom, 0° and 45°):**
   - Side-by-side: ideal sum, blurred sum, ideal max, blurred max for selected angles

6. **Figure 6 — Pre-blur projections (Shepp-Logan absorption):**
   - Sum and max projections at each angle

7. **Figure 7 — Post-blur projections (Shepp-Logan phase/absorption):**
   - Same grid layout from the blurred volume

8. **Figure 8 — Bead phantom phase projections (pre-blur and post-blur):**
   - Grid of sum and max projections

Each figure saved to the output directory with `plt.savefig()` and displayed with `plt.show()`.

## Part 7: Projection Recovery via Deconvolution

This part investigates whether standard projections along Z can be recovered from limited-angle tilted projections, framed as inverse problems analogous to the QPI defocus reconstruction in `QPI_defocus_simulation.py`.

### Problem Statement

Given sum projections of a 3D volume at angles ±15° around the Y-axis, can we recover:
1. The **sum (mean) projection** along Z (0°)?
2. The **max projection** along Z (0°)?

### Mathematical Framework

**Sum projection as a linear operator:**
A sum projection at angle θ is a line integral through the volume — the Radon transform. By the Fourier Slice Theorem, the 2D Fourier transform of the sum projection at angle θ gives a slice through the 3D Fourier transform of the object at that angle. This means:
- The relationship between the 3D object and its sum projections is **linear** and expressible as a transfer function in Fourier space.
- Recovery of the 0° sum projection from ±15° sum projections is a **limited-angle tomography** problem that can be solved via Tikhonov regularization.

**Max projection is nonlinear:**
Max projection is `max(f(x,y,z), axis=z)` — a nonlinear operation. It cannot be written as a convolution or transfer function. Recovery of the 0° max projection from ±15° sum projections is an ill-posed nonlinear inverse problem. We will attempt it with a linearized approximation and document the limitations.

### Experiment Structure

Each experiment uses the bead `[o]` phantom and Shepp-Logan phantom. Four cases total:

#### Case A: Ideal projections (no microscope blur)

**A1. Recover sum projection from tilted sum projections**
1. Compute the ground truth: `proj_0 = volume.sum(axis=0)` (Z sum projection).
2. Compute measurements: `proj_+15 = siddon_project(volume, +15°)` and `proj_-15 = siddon_project(volume, -15°)`.
3. **Forward model:** The tilted sum projections can be related to the 0° projection via a transfer function. Specifically, for a thin slab at each z, a ±15° tilt shifts the slab laterally by `z * tan(θ)`. The sum projection at angle θ is a sheared sum of the volume. In Fourier space this is captured by the projection-slice relationship.
4. **Inverse:** Frame as a Tikhonov-regularized deconvolution:
   - Stack the ±15° projections as a measurement vector.
   - Build the forward operator `H` that maps the 3D object (or its Z-projection) to the tilted projections.
   - Solve: `x_hat = argmin ||Hx - y||^2 + λ||x||^2`
   - Sweep regularization strength and evaluate recovery quality (SSIM, PSNR vs ground truth).

**A2. Recover max projection from tilted sum projections**
1. Ground truth: `max_proj_0 = volume.max(axis=0)`.
2. Same measurements as A1.
3. **Approach:** Since max is nonlinear, use a naive baseline:
   - Reconstruct the 3D volume from the tilted sum projections (limited-angle back-projection or Tikhonov in 3D).
   - Take the max projection of the reconstructed volume.
4. Compare against ground truth max projection. Document the artifacts and limitations.

#### Case B: With microscope blur

Repeat A1 and A2, but now the 3D volume is blurred by the microscope PSF. Projections are computed via Siddon on both the unblurred and blurred volumes (as established in Part 5), so ground truths at both stages are already available.

**B1. Recover sum projection of blurred volume from tilted sum projections of blurred volume**
1. Blur the volume: `blurred_volume = apply_transfer_function(volume, OTF)`.
2. Compute Siddon sum projections of `blurred_volume` at 0° and ±15°.
3. The forward model combines OTF convolution + Siddon projection at angle θ.
4. Two sub-approaches:
   - **Joint deconvolution:** Invert both OTF and projection geometry simultaneously.
   - **Sequential:** First deconvolve each tilted projection (using the projected OTF at that angle), then solve the projection recovery problem.
5. Compare against: (a) sum projection of the *blurred* volume at 0° (post-blur ground truth), (b) sum projection of the *unblurred* volume at 0° (pre-blur ground truth).

**B2. Recover max projection of blurred volume from tilted sum projections of blurred volume**
1. Same blurred volume and tilted measurements as B1.
2. Reconstruct the 3D blurred volume from tilted Siddon sum projections.
3. Apply 3D deconvolution (Tikhonov with the OTF).
4. Take Siddon max projection of the deconvolved volume at 0°.
5. Compare against: (a) max projection of blurred volume (post-blur ground truth), (b) max projection of unblurred volume (pre-blur ground truth).

### Implementation Details

**Forward and adjoint operators using Siddon's algorithm:**
```python
def tilted_sum_projection_operator(volume, angle_deg, voxel_size):
    """Forward: 3D volume -> 2D sum projection at angle via Siddon ray tracing."""
    return siddon_project(volume, angle_deg, voxel_size, mode="sum")

def adjoint_tilted_sum_projection_operator(projection_2d, volume_shape, angle_deg, voxel_size):
    """Adjoint: 2D projection -> 3D volume (back-projection at angle).

    For each ray at the given angle, distribute the detector pixel value
    back into the voxels the ray traverses, weighted by intersection length.
    This is the transpose of the Siddon forward operator.
    """
    volume = np.zeros(volume_shape)
    theta = np.radians(angle_deg)
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    ny, n_lateral = projection_2d.shape

    for iy in range(ny):
        for il in range(n_lateral):
            origin = compute_ray_origin(il, iy, direction, volume_shape, voxel_size)
            voxel_indices, lengths = siddon_trace(origin, direction, volume_shape, voxel_size)
            for k, (iz, _, ix) in enumerate(voxel_indices):
                volume[iz, iy, ix] += lengths[k] * projection_2d[iy, il]
    return volume
```

**Tikhonov solver (iterative, since operators are large):**
Use conjugate gradient on the normal equations `(H^T H + λI) x = H^T y`, where H is the forward operator and y is the stacked tilted projections. Implement with `scipy.sparse.linalg.cg` or a simple gradient descent loop.

**Evaluation metrics:**
- SSIM (structural similarity) between recovered and ground truth projection
- PSNR (peak signal-to-noise ratio)
- Visual comparison plots

### Visualization for Part 7

**Figure 9 — Case A (no blur, pre-blur projections only):**
- Row 1: Ground truth sum projection (0°), tilted sum projections (+15°, -15°), recovered sum projection, difference map
- Row 2: Ground truth max projection (0°), recovered max projection from tilted sums, difference map

**Figure 10 — Case B (with microscope blur):**
- Row 1: Pre-blur ground truth sum projection (0°), post-blur ground truth sum projection (0°), tilted post-blur projections (+15°, -15°)
- Row 2: Jointly recovered sum projection, sequentially recovered sum projection, difference maps vs both ground truths
- Row 3: Pre-blur max projection (0°), post-blur max projection (0°), recovered max projection from deconvolved volume, difference maps

**Figure 11 — Regularization sweep:**
- SSIM/PSNR vs regularization strength λ for each case (A1, A2, B1, B2)

## File Structure

```
docs/examples/demos/projection_modeling/
├── projection_modeling.py          # Main script (sphinx-gallery compatible)
```

## Implementation Order

1. Write configurable parameters section
2. Implement `generate_bead_pattern_phantom()` — returns fluorescence density and phase tensors
3. Implement `generate_shepp_logan_3d()` — returns absorption density tensor
4. Implement `siddon_trace()` and `siddon_project()` (Siddon's ray-tracing algorithm)
5. Implement `compute_projections()` wrapper using Siddon
6. Compute **pre-blur projections** of both phantoms at all angles (sum and max)
7. Run fluorescence forward simulation (OTF blur) for both phantoms
8. Run phase forward simulation for both phantoms
9. Compute **post-blur projections** of blurred volumes at all angles (sum and max)
10. Visualization code for Parts 1–6 (Figures 1–8), including pre-blur vs post-blur comparisons
11. Implement Siddon-based forward/adjoint tilted projection operators
12. Implement Tikhonov solver (CG-based) for projection recovery
13. Case A: Ideal projection recovery experiments (A1 sum, A2 max) using pre-blur projections
14. Case B: Blurred projection recovery experiments (B1 sum, B2 max) with both ground truths
15. Evaluation metrics (SSIM, PSNR) and regularization sweep
16. Visualization code for Part 7 (Figures 9–11)
17. Test the script runs end-to-end
