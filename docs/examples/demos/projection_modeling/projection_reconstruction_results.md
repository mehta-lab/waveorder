# Projection Modeling and Reconstruction

Three phantoms — point bead, line `[o]` pattern, Shepp-Logan — each with
fluorescence and phase channels (256^3, 50 nm isotropic). All
reconstructions use CG-Tikhonov (50 iterations, lambda = 0.001,
ramp-filtered backprojection). PSNR computed after optimal gain calibration
against the unblurred ground-truth object.

## Forward Simulation

1. Generate 3D phantom.
2. Convolve with 3D transfer function (fluorescence OTF or phase TF).
3. Scale to [100, 1024] detector counts; add Poisson noise.
4. Compute Siddon mean-projections at 29 angles (-70 to +70 deg, 5 deg steps).

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

## 1. Limited-Angle Tomography (29 projections, no blur)

Forward model: Siddon ray-tracing at 29 angles (-70 to +70 deg). Source:
unblurred `object`. Each figure shows sinogram, reconstruction, object,
difference, and kz-kx Fourier cross-sections.

### Point

![](plots/point/point_geometric.png)

### Lines [o]

![](plots/lines/lines_geometric.png)

### Shepp-Logan

![](plots/shepplogan/shepplogan_geometric.png)

The ±70 deg range leaves a missing cone of half-angle 20 deg along the
optical axis. XZ slices show axial elongation; Fourier panels confirm
structured banding where the ground truth has uniform support. PSNR
degrades with phantom complexity: the sparse point bead reconstructs well,
while the Shepp-Logan phantom's overlapping low-contrast ellipsoids make
the inverse problem ill-conditioned.

---

## 2. Two-View Tomography (2 projections, no blur)

Forward model: Siddon ray-tracing with two projections at ±theta. Source:
unblurred `object`. Theta swept from 5 to 70 deg in 5 deg steps.

### PSNR vs Projection Half-Angle

![](plots/psnr_vs_theta_geometric_two_projections.png)



### Reconstructions at Optimal Angle

#### Point (±10 deg)

![](plots/point/point_geometric_two_projections.png)

#### Lines [o] (±70 deg)

![](plots/lines/lines_geometric_two_projections.png)

#### Shepp-Logan (±45 deg)

![](plots/shepplogan/shepplogan_geometric_two_projections.png)

The optimal angle depends on phantom structure. The point bead is spectrally
compact — PSNR is flat across all thetas (45.3-45.4 dB). The lines pattern
benefits from wide-angle projections (monotonically increasing, peak at
±70 deg). Shepp-Logan peaks at ±45 deg, where two projections maximally
separate the overlapping ellipsoids.

Two views lose 3-5 dB relative to 29 projections. The gap is smallest for
the sparse point bead (3.6 dB) and largest for the extended Shepp-Logan
(5.4 dB). Two projections sample only two Fourier slices; the XZ FFT panels
show energy along two narrow bands with the rest near zero.

---

## 3. Limited-Angle Tomography with Blur (29 projections, OTF)

Forward model: 3D OTF convolution + Siddon ray-tracing at 29 angles. Source:
blurred + noisy `rawimage`. By the Fourier-slice theorem, the 3D OTF
convolution before projection applies the angle-dependent 2D transfer
function at each tilt.

### PSNR

| Sample | Fluorescence | Phase |
|--------|-------------|-------|
| point | 45.86 dB | 45.37 dB |
| lines | 30.25 dB | 30.35 dB |
| shepplogan | 16.13 dB | 14.57 dB |

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

The wave model reconstructs worse than the geometric model. Three factors
compound: (1) the OTF zeros high frequencies that Tikhonov regularization
cannot recover; (2) Poisson noise fills the spectrum beyond the OTF
passband; (3) the gain mismatch between detector counts and physical units
(alpha_wave ~ 4e-5 vs alpha_geo ~ 1.0). The phase channel loses an
additional 1-3 dB because its transfer function has a null at kz = 0.

---

## 4. Two-View Tomography with Blur (2 projections, OTF)

*Not yet computed.* The `wave-two-projections` subcommand exists; results
will follow the same sweep format as Section 2.

---

## Algorithms

### Forward models

**Geometric:** `H(x) = [siddon_project(x, theta) for theta in angles]`.
**Wave-optical:** `H(x) = [siddon_project(OTF * x, theta) for theta in angles]`.

The 3D OTF convolution (FFT on GPU) precedes Siddon projection. The adjoint
reverses both: `H^T(y) = conj(OTF) * sum(siddon_backproject(y_i, theta_i))`.

### Siddon ray-tracing

Each projection computes line integrals through a 3D volume at tilt angle
theta (rotation about Y in the ZX plane). Ray geometry is precomputed as a
sparse matrix A of shape (n_lateral, nz * nx). Forward projection:
`torch.sparse.mm(A, vol_zx)`; backprojection: `torch.sparse.mm(A.T, proj.T)`.

### Conjugate Gradient with Tikhonov regularization

Both models solve `(H*H + lambda I)x = H*y` via CG. Ramp-filtered
backprojection preconditions the normal operator so all frequencies converge
at a similar rate.

![](plots/drawings/operator_adjoint_spaces.png)

![](plots/drawings/normal_equation_derivation.png)

### Gain calibration

Before computing PSNR, the optimal gain `alpha = <recon, gt> / <recon, recon>`
removes the arbitrary scale between reconstruction and ground truth.
