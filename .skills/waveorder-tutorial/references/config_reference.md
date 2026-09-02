# waveorder config reference

A reconstruction config is a YAML file: top-level keys plus one of the
`phase:` / `fluorescence:` blocks. Copy a starting template from the repo:
`docs/examples/cli/configs/{phase,fluorescence}_{2d,3d}.yml`.

## Top level

```yaml
input_channel_names:        # list; must match channel names in the OME-Zarr
- Brightfield
time_indices: all           # "all", an int, or a list of ints
reconstruction_dimension: 3 # 2 for thin samples, 3 for thick
device: null                # null=cpu, "auto", "cuda:0", "mps"
```

Exactly one of `phase:` or `fluorescence:` for a first reconstruction
(they cannot coexist in one file; birefringence is separate).

## `phase:` block

```yaml
phase:
  transfer_function:
    wavelength_illumination: 0.532       # µm
    yx_pixel_size: 0.1                   # µm
    z_pixel_size: 0.25                   # µm
    z_padding: 0                         # z slices padded for axial boundaries
    z_focus_offset: 0                    # (sweepable/optimizable) focus slice offset
    index_of_refraction_media: 1.3       # 1.0 air / 1.33 water / 1.47 oil
    numerical_aperture_detection: 1.2    # objective NA (must be <= media RI)
    numerical_aperture_illumination: 0.9 # condenser NA (must be <= media RI)
    tilt_angle_zenith: 0.0               # radians; keep 0 unless shadow-cast contrast
    tilt_angle_azimuth: 0.0              # radians; shadow direction
    invert_phase_contrast: false         # TRY BOTH; see density-sign test below
  apply_inverse:
    reconstruction_algorithm: Tikhonov   # Tikhonov (fast) or TV (edge-preserving)
    regularization_strength: 0.001       # main noise/sharpness dial
    TV_rho_strength: 0.001               # TV only: ADMM rho
    TV_iterations: 1                     # TV only: ADMM iterations
```

### `invert_phase_contrast`

The hardest knob. It sets the sign convention of density contrast and depends
on the microscope. Reconstruct with both `true` and `false`, then pick the one
where objects denser than the surrounding media appear brighter than
background:

- **more-dense** regions (nucleoli, membranes, cytoplasm, dense organelles)
  should appear **brighter** than background
- **less-dense** regions (nuclei overall, vacuoles, lumen) should appear
  **darker** than background

### tilt angles

Leave at `0.0` for drafts. Add a nonzero `tilt_angle_zenith` only if the
in-focus raw data visibly shows one-sided, DIC-like "shadow-cast" contrast;
set `tilt_angle_azimuth` to the shadow direction. Large tilts usually need
several rounds of azimuth/zenith refinement; small tilts show up as a point
spread function that drifts sideways through focus.

## `fluorescence:` block

Same optical keys, but with emission instead of illumination:

```yaml
fluorescence:
  transfer_function:
    yx_pixel_size: 0.1
    z_pixel_size: 0.25
    z_padding: 0
    z_focus_offset: 0
    index_of_refraction_media: 1.3
    numerical_aperture_detection: 1.2
    tilt_angle_zenith: 0.0
    tilt_angle_azimuth: 0.0
    wavelength_emission: 0.532           # µm (emission, not excitation)
    confocal_pinhole_diameter: null      # null = widefield; else diameter
  apply_inverse:
    reconstruction_algorithm: Tikhonov   # Tikhonov, TV, RL, RLGC
    regularization_strength: 0.001
    TV_rho_strength: 0.001
    TV_iterations: 1
```

`RL` (Richardson-Lucy) and `RLGC` (RL with automatic stopping) are available
for 3D fluorescence deconvolution; they take an `rl:` sub-block for iteration
settings. Start with `Tikhonov`, then trial RL/RLGC if the user wants sharper
deconvolution.

## Which knobs to sweep

| symptom                        | knob                                        |
|--------------------------------|---------------------------------------------|
| grainy / noisy                 | raise `regularization_strength`             |
| over-smoothed / lost detail    | lower `regularization_strength`             |
| ringing / halos at edges       | sweep `numerical_aperture_illumination`; try `TV` |
| wrong density sign (phase)     | flip `invert_phase_contrast`                |
| 2D output slightly defocused   | sweep `z_focus_offset`                      |
| top/bottom Z slices similar    | increase `z_padding`                        |
| PSF drifts sideways vs focus   | sweep `tilt_angle_azimuth` + `tilt_angle_zenith` together |
| shadow-cast in-focus contrast  | large tilt; iterate azimuth then zenith     |

`regularization_strength` is best swept over decades (1e-4 ... 1e0). Pick the
"elbow": the smallest value that removes noise/ringing without softening real
structures.

## Auto-tuning (`optimization:` block)

Any of `z_focus_offset`, `tilt_angle_azimuth`, `tilt_angle_zenith`,
`numerical_aperture_illumination`, and `numerical_aperture_detection` can be
made optimizable by replacing the float with an `init`/`lr` pair:

```yaml
phase:
  transfer_function:
    z_focus_offset:
      init: 0
      lr: 0.1
optimization:                 # optional; these are sensible starting values
  max_iterations: 10
  method: adam                # adam, lbfgs, nelder_mead, grid_search
  loss:
    type: midband_power       # also: total_variation, laplacian_variance,
                              # normalized_variance, spectral_flatness
  log_dir: null               # set a path for TensorBoard logs
```

When `wo rec` sees any optimizable parameter, it optimizes on the first input
position, writes `<config>_optimized.yml` next to the config with the fitted
values, and then reconstructs with them. Tune one parameter at a time and
check the fitted value against the user's manually chosen value before
trusting it. A 2D config with an optimizable `z_focus_offset` is waveorder's
autofocus.

Full working example: `docs/examples/optimization/phase_2d_optimized.yml`.
