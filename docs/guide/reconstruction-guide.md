# Automating reconstructions

`waveorder` provides a configuration-file-based command-line interface (CLI) for file-based reconstruction. The separate tensor interface below provides opt-in optimized 3D phase reconstruction.

## Preparing your data

`waveorder` is compatible with OME-Zarr, a chunked next generation file format with an [open specification](https://ngff.openmicroscopy.org/0.4/). All acquisitions completed with the `waveorder` plugin will be automatically converted to `.zarr` format, and existing data can be converted using `iohub`'s `convert` utility.

Inside a `waveorder` environment, convert a Micro-Manager TIFF sequence, OME-TIFF, or pycromanager NDTiff dataset with
```
iohub convert `
    -i ./dataset/ `
    -o ./dataset.zarr
```

## How can I use `waveorder`'s CLI to perform reconstructions?
`waveorder`'s CLI is summarized in the following figure:
<img src="../_static/images/cli_structure.png" align="center">

The main command `waveorder` command is composed of two subcommands: `compute-tf` and `apply-inv-tf`.

A reconstruction can be performed with a single `reconstruct` call. For example:
```
waveorder reconstruct `
    -i ./data.zarr/*/*/* `
    -c ./config.yml `
    -o ./reconstruction.zarr
```
Equivalently, a reconstruction can be performed with a `compute-tf` call followed by an `apply-inv-tf` call. For example:
```
waveorder compute-tf `
    -i ./data.zarr/0/0/0 `
    -c ./config.yml `
    -o ./tf.zarr

waveorder apply-inv-tf
    -i ./data.zarr/*/*/* `
    -t ./tf.zarr `
    -c ./config.yml `
    -o ./reconstruction.zarr
```
Computing the transfer function is typically the most expensive part of the reconstruction, so saving a transfer function then applying it to many datasets can save time.

## Input options

The input `-i` flag always accepts a list of inputs, either explicitly e.g. `-i ./data.zarr/A/1/0 ./data.zarr/A/2/0` or through wildcards `-i ./data.zarr/*/*/*`. The positions in a high-content screening `.zarr` store are organized into `/row/col/fov` folders, so `./input.zarr/*/*/*` creates a list of all positions in a dataset.

The `waveorder compute-tf` command accepts a list of inputs, but it only computes the transfer function for the first position in the list. The `apply-inv-tf` command accepts a list of inputs and applies the same transfer function to all of the inputs, which requires that all positions contain arrays with matching TCZYX shapes.

## What types of reconstructions are supported?
See `/waveorder/examples/` for a list of example configuration files.

WIP: This documentation will be expanded for each reconstruction type and parameter.

## Opt-in optimized 3D phase reconstruction

`waveorder.reconstruction.PhaseReconstruction` provides two implementations behind one callable interface. The existing model, xarray and CLI interfaces remain unchanged and available.

```python
import torch
from waveorder.reconstruction import PhaseReconstruction

# data and next_volume are contiguous float32 ZYX tensors on CUDA.
with PhaseReconstruction(
    tuple(data.shape),
    yx_pixel_size=0.2,
    z_pixel_size=0.5,
    wavelength_illumination=0.532,
    z_padding=20,
    index_of_refraction_media=1.33,
    numerical_aperture_illumination=0.6,
    numerical_aperture_detection=1.0,
    regularization_strength=0.01,
    backend="cuda",
    device=data.device,
) as reconstruct:
    phase = reconstruct(data)
    next_phase = reconstruct(next_volume)
    # phase remains valid after the second call and after the context closes.
    destination = torch.empty_like(data)
    reconstruct(next_volume, out=destination)
```

Use `backend="torch"` for the optimized PyTorch implementation on CPU or CUDA. Use `backend="cuda"` for the optional native Linux implementation, which requires CUDA-enabled PyTorch, a compatible CUDA toolkit and C++ compiler, and Ninja from `waveorder[native]`. Native compilation happens only when explicitly constructing the CUDA implementation, not when importing the namespace. `TORCH_EXTENSIONS_DIR` can place build files on suitable scratch storage. Loading failures are reported rather than switching backends.

Both implementations reconstruct one full float32 ZYX volume with Tikhonov regularization and no apodization. They preserve the complete transverse and axial Fourier domains. Optical slabs limit intermediate memory; they do not divide the specimen into independently reconstructed tiles. Results are float32 phase in cycles per voxel on the input device. Transfer to CPU explicitly with `phase.cpu()` when needed. Float64, batched volumes and apodization remain available through the existing interfaces; this new interface does not silently convert those requests.

Regularization must be a finite real scalar between zero and the float32 maximum, checked before conversion. Zero preserves singularities in the reference inverse rather than adding an epsilon. `absorption_ratio=None` omits absorption; a supplied ratio, including a learnable zero tensor, retains it.

The Torch implementation supports input, NA, tilt, regularization and absorption-ratio gradients. Scalar tensor parameters passed to the constructor remain live: each call reads their current values and creates a fresh optical/filter graph, so in-place optimizer updates are supported. Fixed Python scalar settings cache one filter while still allowing fresh input-gradient graphs. Sampling settings are fixed. Small-case gradient checks do not imply that a full-volume backward pass fits in memory.

The native implementation is inference-only. It snapshots fixed no-gradient parameters and rejects inputs or parameters that require gradients. Its optical filter is currently prepared with bounded Torch operations; the solve uses reusable cuFFT plans and native reduction, packing, filtering and output kernels. This is separate from the existing CUDA optical-fitting implementation.

Default outputs have stable caller-owned storage. `out=` explicitly reuses matching contiguous float32 output storage and rejects gradient-bearing tensors or input/output overlap. Native output kernels write directly into that destination without cloning a Plan-owned result. Finish reading an output before reusing it as `out`. When producing inputs or parameters on another CUDA stream, establish that dependency before the call. The reconstruction object orders its private filter and workspace operations.

Each object owns one prepared configuration and releases it on `close()` or context exit. Create a new object when fixed geometry, device or settings change. There is no hidden global GPU cache. Half-precision file storage or transport is a separate data-loading policy; this interface never quantizes float32 input to half.

## Opt-in optimized 3D fluorescence deconvolution

`WienerButterworthRL` and `GradientConsensusRL` accept a precomputed, unshifted
complex64 OTF with shape `(Z + 2*z_padding, Y, X)` and reconstruct one contiguous
float32 ZYX tile per call. Each object reuses its prepared filters and FFT
workspace across calls. These are opt-in tensor interfaces; the existing
fluorescence model and CLI remain available.

```python
import torch
from waveorder.reconstruction import GradientConsensusRL, WienerButterworthRL

# otf is the physical fluorescence OTF, already on volume.device.
with WienerButterworthRL(
    otf, z_padding=20, alpha=0.005, beta=0.005, order=6,
    iterations=1, backend="cuda", device=volume.device,
) as solver:
    rl_density = solver(volume)

with GradientConsensusRL(
    otf, z_padding=20, iterations=25, backend="cuda", device=volume.device,
) as solver:
    rng = torch.Generator(device=volume.device).manual_seed(42)
    rl_gc = solver(volume, generator=rng)
```

Both have optimized Torch (`backend="torch"`, CPU or CUDA) and optional native
CUDA inference paths. Native CUDA builds explicitly using the matching CUDA
toolkit, a C++17 compiler, Ninja and `TORCH_EXTENSIONS_DIR` on suitable scratch.
It never falls back silently. Outputs remain valid after another call or
`close()`; use `out=` only with non-overlapping caller-owned float32 storage.

The unmatched Wiener–Butterworth projector belongs to RL only. Its default
`beta_convention="reference"` matches the existing Guo reference convention;
one iteration is the usual choice, and more than five warns about artifacts.
If an undersampled PSF prevents FWHM estimation, set
`resolution_mode="manual"` and supply `resolution_zyx_px`. RL-GC always uses
the matched adjoint, consumes the provided Torch generator for its photon
split, and can stop early when the consensus freezes every voxel. Small FFT
roundoff can change a few near-zero consensus decisions, so seeded outputs
need not be bitwise identical to the full-complex reference. Keep the OTF,
padding, sampling, photon-count scale and iteration count fixed when comparing
implementations. These operators do not load or spatially tile a whole dataset
for you; reuse one object across identically shaped full-Z tiles.

### Full-Z inference comparison

One H100, synthetic Gaussian PSF/OTF and Poisson counts with mean 8, input
`2372 × 185 × 1024`, 50-plane Z padding, float32 inference. Wiener–Butterworth
RL used one iteration, `alpha=beta=0.005`, order 6, and manual resolution
`(4, 3, 3)` pixels; RL-GC used three iterations with a fixed generator seed.
Times are synchronized wall-clock medians after two warmups and over five calls
with a preallocated `out=` buffer. They exclude OTF construction, first-use
setup, extension compilation, file I/O and device transfers. Peaks count Torch
allocator memory, including the resident input, filters and output, not all
device usage.

| Algorithm | Optimized Torch | Native CUDA |
| --- | ---: | ---: |
| Wiener–Butterworth RL, 1 iteration [ms] | 51.49 | 30.14 |
| Wiener–Butterworth RL, peak [GB] | 20.472 | 16.722 |
| RL-GC, 3 iterations [ms] | 667.80 | 426.70 |
| RL-GC, peak [GB] | 33.588 | 27.958 |

Wiener–Butterworth RL uses H's DC coefficient for its first rate and computes
padding, clamping and the initial ratio in one row-indexed kernel. RL-GC
uses `|H|²` for its crosstalk convolution, reuses two real volumes after
their last read, and writes out-of-place cuFFT results directly into those
volumes. Its seeded native result differed from optimized Torch at 879 of
449,351,680 voxels beyond `rtol=6e-4, atol=5e-5`, with global relative
L2 error `3.75e-5`. Nearly-zero consensus decisions can flip under FFT
roundoff.

Against the original full-complex implementations at the **same** OTF,
padding and RL-GC photon-split seed, the 50-plane native Wiener result had
relative L2 error `2.26e-7`, maximum absolute error `1.05e-5`, and no voxels
outside that tolerance. Native RL-GC had relative L2 error `4.41e-5`;
1,185 of 449,351,680 voxels crossed tolerance, with maximum absolute
difference `0.679` where consensus freeze decisions differed. At 64 planes
with both paths using the newly sampled OTF, the respective relative L2
errors were `1.47e-7` (Wiener, zero failures) and `2.73e-5` (RL-GC,
486 failures). These are parity checks, not microscope-data quality tests.

Rebuilding the OTF with 64-plane Z padding makes the same input's FFT length
2500 instead of 2472. Native Wiener–Butterworth RL took 20.93 ms and peaked
at 16.871 GB; native RL-GC took 324.63 ms and peaked at 28.234 GB. On this
Gaussian OTF, the Wiener–Butterworth results for 50 and 64 planes agreed at
every output voxel within the tolerance above (relative L2 `2.66e-7`).
For RL-GC, a fixed seed does not preserve photon assignments when the padded
tensor shape changes. Holding the interior photon splits fixed in a diagnostic
comparison gave relative L2 `4.58e-5`, with 1,216 voxels outside tolerance.
That controlled split reuses reflected halo assignments and is not the normal
stochastic run.

A synthetic 192-wide Y tile with 64-plane Z padding took 16.06 ms for
Wiener–Butterworth RL and 275.61 ms for RL-GC. Extra Y reflection is **not**
an interchangeable speed setting: reconstructing the same original input
after reflecting seven Y planes changed the whole-volume Wiener result by
relative L2 `5.63e-3` and the controlled-split RL-GC result by `3.57e-3`.
After excluding 16 Y pixels at each edge and 100 Z planes at each end,
relative L2 fell to `2.74e-5` and `4.46e-5` respectively. A 192-wide
extraction therefore needs a validated overlap/crop policy, with timing
measured per **retained** pixel, not just per tile.

Neither geometry has been validated with a physical microscope OTF or real
fluorescence data. Choose tiling and halo size from the optics and boundary
requirements, then measure whole-tile throughput and reconstruction quality;
the solvers cannot silently change the shape of a supplied OTF.

### Real-fluorescence parity spot check

One H100, a 488 nm Side A channel in a zebrafish OME-Zarr, with the optical
axis moved to ZYX before reconstruction: `2404 × 96 × 192` voxels, full
acquisition depth. The OTF was generated by WaveOrder's physical model using
the recorded lateral/axial spacing (`0.1842`/`1.018` µm) and **assumed**
532 nm emission, refractive index 1.3 and detection NA 1.1. No measured PSF
or calibrated OTF was available. Both solvers used the same 50-plane halo;
Wiener–Butterworth used one iteration, alpha/beta 0.005, order 6 and manual
resolution `(4, 3, 3)` pixels, while RL-GC used three iterations and the
same photon-split seed in both implementations.

| Solver | Original full-complex [ms] | Native CUDA [ms] | Original/native peak [GB] | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Wiener–Butterworth RL | 16.20 | 2.62 | 2.763 / 1.469 | `3.59e-7` |
| RL-GC | 103.57 | 42.79 | 4.240 / 2.576 | `2.40e-7` |

All 44,310,528 output voxels were within `rtol=6e-4, atol=5e-5` for both
solvers; the RL-GC generator ended in the same state. Times are synchronized
medians of five calls after two warmups, excluding the 0.83 s read and 9.19 s
model-OTF build. This checks software parity on real input, not optical
calibration or biological ground truth.
