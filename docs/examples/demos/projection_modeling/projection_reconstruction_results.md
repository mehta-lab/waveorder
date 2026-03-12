# Projection Reconstruction Results

CG-Tikhonov reconstruction (50 iterations, λ = 0.001) from 29 Siddon
projections at −70° to +70° in 5° steps.  PSNR is computed after
optimal gain correction against the unblurred ground-truth `object`.

## Geometric reconstruction

Forward model: Siddon ray-tracing only.  Source: unblurred `object` column.

| Sample     | Fluorescence PSNR | Phase PSNR |
|------------|-------------------|------------|
| point      | 48.98 dB          | 48.98 dB   |
| lines      | 33.78 dB          | 33.78 dB   |
| shepplogan | 21.05 dB          | 21.05 dB   |

## Wave-optical reconstruction

Forward model: 3D OTF convolution + Siddon projection.  Source:
blurred + noisy `rawimage` column (BLACK_LEVEL subtracted before
projection).

| Sample     | Fluorescence PSNR | Phase PSNR |
|------------|-------------------|------------|
| point      | 46.26 dB          | 45.13 dB   |
| lines      | 30.83 dB          | 29.61 dB   |
| shepplogan | 16.94 dB          | 14.28 dB   |

## Observations

1. **Geometric > wave by 2–6 dB.**  Expected: the wave model must
   simultaneously deconvolve OTF blur and suppress Poisson noise.

2. **Simpler phantoms reconstruct better.**  The point bead is sparse
   and well-conditioned; the Shepp-Logan phantom has broad,
   low-contrast features that overlap across projection angles.

3. **Phase and fluorescence track closely** within each model, because
   both channels share the same projection geometry and differ only in
   their transfer functions.

## Bugs fixed during development

1. **Padding misalignment.**  Center-padding projections to uniform
   width shifted the data, but `siddon_backproject` read from
   unshifted indices.  The adjoint operator received zeros instead of
   projection values, causing divergence (shepplogan PSNR = −79.69 dB
   before the fix).  Removed padding entirely; the CG solver works
   with variable-width projection lists.

2. **Scale mismatch.**  The `rawimage` column stores data scaled to
   [100, 1024] with Poisson noise.  The OTF-based forward model
   operates in physical units.  Fixed by subtracting BLACK_LEVEL
   before projecting and applying optimal gain correction when
   computing metrics.
