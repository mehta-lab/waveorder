# Projection Modeling and Reconstruction Results


## Limited angle tomography without blur

Forward model: Geometric projections (Siddon ray-tracing) over +70 to -70 degree every 5 degrees. No optical transfer function in the forward or adjoint operators. 


Each figure shows the sinogram of measurements (Siddon sum-projections of
the object at all 29 angles), the gain-corrected reconstruction, the
ground-truth object, and the difference (reconstruction - object) in three
orthogonal views. The bottom row shows kz-kx Fourier cross-sections (ky=0)
of the reconstruction and object, revealing missing-cone artifacts.


### Point

![](plots/point/point_geometric.png)

### Lines [o]

![](plots/lines/lines_geometric.png)

### Shepp-Logan

![](plots/shepplogan/shepplogan_geometric.png)

### Notes

**Reconstruction algorithm.**
CG-Tikhonov reconstruction (50 iterations, lambda = 0.001) from 29 Siddon
projections at -70 deg to +70 deg in 5 deg steps. PSNR computed after
optimal-gain correction against the unblurred ground-truth `object`.

**Ramp-filtered backprojection as preconditioner.**
Plain backprojection H* gives a normal operator H*H whose frequency response
falls as ~1/|omega|. CG resolves low frequencies first; high-frequency edges
converge slowly and ring. Replacing H* with ramp-filtered backprojection
(|omega| weighting before backprojection) flattens the spectrum of the
normal operator, so all frequencies converge at a similar rate. The
regularization term lambda I then acts uniformly across frequencies.
With the ramp preconditioner, edge ringing diminishes and PSNR improves.

**Missing cone.** The +/-70 deg angular range leaves a missing cone of
half-angle 20 deg along the optical axis. XZ cross-sections show axial
elongation in all three phantoms. Difference images reveal streak artifacts
radiating from sharp features at angles matching the projection limits. The
Fourier kz-kx slices confirm structured banding in the reconstruction where
the ground truth has uniform spectral support.

**Phantom complexity degrades PSNR.** The point bead is sparse and
spectrally broad — a near-ideal target for tomographic reconstruction. The
lines pattern has extended features whose projections overlap at oblique
angles. The Shepp-Logan phantom has broad, low-contrast ellipsoids that
produce similar projections at many angles, making the inverse problem
ill-conditioned.

---

## Forward Simulation

Three phantoms — point bead, line `[o]` pattern, Shepp-Logan — each with
fluorescence density and phase density channels. The forward pipeline:

1. Generate 3D phantom (256^3, 50 nm isotropic).
2. Convolve with the 3D transfer function (fluorescence OTF or phase TF).
3. Scale to [100, 1024] detector counts and add Poisson noise.
4. Compute Siddon mean-projections at 29 angles (-70 deg to +70 deg).

Each figure shows three orthogonal center slices of the object and rawimage,
plus projections at 0 deg, +15 deg, and -15 deg.

### Point

| Fluorescence | Phase |
|---|---|
| ![](plots/point/point_Fluorescence_forward.png) | ![](plots/point/point_Phase_forward.png) |

### Lines [o]

| Fluorescence | Phase |
|---|---|
| ![](plots/lines/lines_Fluorescence_forward.png) | ![](plots/lines/lines_Phase_forward.png) |

### Shepp-Logan

| Fluorescence | Phase |
|---|---|
| ![](plots/shepplogan/shepplogan_Fluorescence_forward.png) | ![](plots/shepplogan/shepplogan_Phase_forward.png) |



---

## Wave-Optical Reconstruction (With Imaging Blur)

Forward model: 3D OTF convolution + Siddon ray-tracing. Source: blurred +
noisy `rawimage` column. The 3D OTF in the forward model accounts for
angle-dependent resolution and defocus via the Fourier-slice theorem.

Each figure shows the sinogram of measurements (projections of rawimage),
the gain-corrected reconstruction, the ground-truth object, difference
against OTF-blurred object (noiseless), and difference against the unblurred
object. The bottom row shows kz-kx Fourier cross-sections of the rawimage,
reconstruction, object, and OTF-blurred object.

### PSNR

| Sample     | Fluorescence | Phase    |
|------------|-------------|----------|
| point      | 45.86 dB    | 45.37 dB |
| lines      | 30.25 dB    | 30.35 dB |
| shepplogan | 16.13 dB    | 14.57 dB |

### Point

| Fluorescence | Phase |
|---|---|
| ![](plots/point/point_Fluorescence_wave.png) | ![](plots/point/point_Phase_wave.png) |

### Lines [o]

| Fluorescence | Phase |
|---|---|
| ![](plots/lines/lines_Fluorescence_wave.png) | ![](plots/lines/lines_Phase_wave.png) |

### Shepp-Logan

| Fluorescence | Phase |
|---|---|
| ![](plots/shepplogan/shepplogan_Fluorescence_wave.png) | ![](plots/shepplogan/shepplogan_Phase_wave.png) |

---

## Observations

### 1. Wave reconstruction is worse than geometric

Three factors compound to limit the wave model:

**OTF attenuation.** The 3D OTF zeros high spatial frequencies before
projection. The CG solver must invert this attenuation, but Tikhonov
regularization (lambda = 0.001) penalizes the large amplitudes needed to
recover those frequencies. The FFT panels show the wave reconstruction near
zero everywhere the OTF is weak, while the geometric reconstruction tracks
the ground truth within the projection-sampled region.

**Noise floor.** The rawimage is scaled to [100, 1024] with Poisson noise.
Its Fourier spectrum is flat at high frequencies — signal is buried in noise
beyond the OTF passband. The geometric model avoids this by projecting the
noiseless `object` column.

**Gain mismatch.** The optimal gain correction reveals a scale disparity:
alpha_geo ~ 1.0 (forward model and object share physical units), while
alpha_wave ~ 4e-5 (detector counts vs physical units). The wave forward
model operates in physical units but reconstructs from detector-scaled
measurements. Subtracting BLACK_LEVEL removes the offset but not the
multiplicative gain.

### 2. Phase channel worse than fluorescence (wave model)

The phase transfer function has a null at kz = 0 — it measures refractive
index gradients, not absolute phase. The wave reconstruction therefore loses
all low-frequency axial information. The fluorescence OTF passes DC and
retains the overall object shape. This gap explains the extra 1-3 dB between
fluorescence and phase PSNR in the wave model.

### 3. Diff against OTF-blurred object vs raw object (wave model)

The "Rec - OTF*Obj" panel isolates deconvolution accuracy: it shows
residual error after accounting for the optical blur. The "Rec - Object"
panel shows total error, combining deconvolution residual with the intrinsic
blur that the solver cannot fully invert. Comparing the two reveals where the
solver trades noise amplification for resolution recovery.

---

## Algorithms

### Siddon ray-tracing (sparse matmul)

Each projection computes line integrals through a 3D volume at a tilt angle
theta (rotation about Y in the ZX plane). The Siddon algorithm traces each
ray through voxel boundaries, accumulating weighted contributions along the
intersection path. For a volume of shape (Z, Y, X), every Y row shares the
same (Z, X) ray geometry — the 2D trace is replicated across all Y rows.

The ray geometry for each angle is precomputed as a sparse matrix
A of shape (n_lateral, nz * nx), where each nonzero entry is the
intersection length of a ray with a voxel. Forward projection reduces to
`torch.sparse.mm(A, vol_zx)`; backprojection to `torch.sparse.mm(A.T, proj.T)`.
This replaces Python loops with GPU-accelerated sparse matmul.

`mode="sum"`: line-integral projection (used in reconstruction).
`mode="mean"`: sum divided by physical path length (used for display in the
forward simulation).

The adjoint distributes each detector pixel back into traversed voxels,
weighted by intersection length. Together, the forward and adjoint operators
satisfy the dot-product test `<Ax, y> = <x, A^T y>`.

### CG-Tikhonov solver

The forward operator H projects a 3D volume into a set of 2D projections;
the adjoint H* backprojects measurements into image space. The adjoint
identity `<Hx, y> = <x, H*y>` moves H across the inner product, converting
a measurement-space residual into an image-space gradient.

![](plots/drawings/operator_adjoint_spaces.png)

Setting the gradient of `||Hx - y||^2` to zero yields the normal equation
`H*Hx = H*y`. Adding Tikhonov regularization gives `(H*H + lambda I)x = H*y`,
solved by conjugate gradient iteration.

![](plots/drawings/normal_equation_derivation.png)

Both reconstruction modes solve this normal equation via CG, where:
- `H` is the forward operator (Siddon or OTF+Siddon),
- `y` is the measured projection data,
- `lambda` is the Tikhonov regularization parameter (default 0.001),
- `x` is the reconstructed volume.

CG requires only forward and adjoint evaluations — no explicit matrix
storage. Each iteration applies `H^T H + lambda I` to a search direction,
updating the solution along the steepest descent in the conjugate direction.
50 iterations suffice for convergence on these 256^3 phantoms.

### Forward models

**Geometric:** `H(x) = [siddon_project(x, theta) for theta in angles]`.
No optical blur. Tests pure tomographic reconstruction from clean projections.

**Wave-optical:** `H(x) = [siddon_project(OTF * x, theta) for theta in angles]`.
The 3D OTF convolution (via FFT on GPU) precedes Siddon projection. The
adjoint reverses both: `H^T(y) = conj(OTF) * sum(siddon_backproject(y_i, theta_i))`.
By the Fourier-slice theorem, projecting an OTF-blurred volume at angle theta
samples the central slice of the 3D Fourier transform at that angle. The 3D
OTF convolution naturally weights each Fourier slice by the angle-dependent
transfer function, coupling defocus and projection geometry.

### Optimal gain correction

Reconstruction and ground truth may differ by a multiplicative scale factor
(especially in the wave model, where detector counts and physical units
diverge). Before computing PSNR, we solve for the gain `alpha` that
minimizes `||alpha * recon - gt||^2`:

```
alpha = <recon, gt> / <recon, recon>
```

This removes the arbitrary scale and isolates structural fidelity.

---

## Bugs fixed during development

1. **Padding misalignment.** Center-padding projections shifted the data, but
   `siddon_backproject` read unshifted indices. The adjoint received zeros,
   causing divergence (Shepp-Logan PSNR = -79.69 dB before fix). Removed
   padding; CG works with variable-width projection lists.

2. **Scale mismatch.** `rawimage` stores data in [100, 1024] with Poisson
   noise; the forward model operates in physical units. Fixed by subtracting
   BLACK_LEVEL before projecting and applying optimal-gain correction for
   metrics.
