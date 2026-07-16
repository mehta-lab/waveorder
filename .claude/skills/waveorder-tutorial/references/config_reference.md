# waveorder config reference

A reconstruction config is a YAML file. Top-level keys plus one of the
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
    z_focus_offset: 0                    # (sweepable) focus slice offset
    index_of_refraction_media: 1.3       # 1.0 air / 1.33 water / 1.47 oil
    numerical_aperture_detection: 1.2    # objective NA
    numerical_aperture_illumination: 0.9 # condenser NA (≤ index_of_refraction_media)
    tilt_angle_zenith: 0.0               # radians; keep 0 unless shadow-cast contrast
    tilt_angle_azimuth: 0.0              # radians; shadow direction
    invert_phase_contrast: false         # TRY BOTH — see density-sign test below
  apply_inverse:
    reconstruction_algorithm: Tikhonov   # Tikhonov (fast) or TV (edge-preserving)
    regularization_strength: 0.001       # main noise/sharpness dial
    TV_rho_strength: 0.001               # TV only: ADMM rho
    TV_iterations: 1                     # TV only: ADMM iterations
```

### `invert_phase_contrast`
The hardest knob. It sets the sign convention of density contrast and depends on
the microscope. Reconstruct with both `true` and `false`, then pick the one
where:
- **less-dense** regions (nuclei, vacuoles, lumen) are **darker** than background
- **more-dense** regions (cytoplasm, membranes) are **brighter** than background

### tilt angles
Leave at `0.0`. Only add a nonzero `tilt_angle_zenith` if the in-focus raw data
visibly shows one-sided, DIC-like "gradient / shadow-casting" contrast; set
`tilt_angle_azimuth` to the shadow direction.

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
    wavelength_emission: 0.532           # µm
    confocal_pinhole_diameter: null      # null = widefield; else diameter (AU/µm)
  apply_inverse:
    reconstruction_algorithm: Tikhonov
    regularization_strength: 0.001
    TV_rho_strength: 0.001
    TV_iterations: 1
```

## Which knobs to sweep

| symptom                     | knob                                     |
|-----------------------------|------------------------------------------|
| grainy / noisy              | ↑ `regularization_strength`              |
| over-smoothed / lost detail | ↓ `regularization_strength`              |
| ringing / halos at edges    | try `TV`; tune `TV_rho_strength`         |
| wrong density sign (phase)  | flip `invert_phase_contrast`             |
| slightly defocused output   | sweep `z_focus_offset`                   |

`regularization_strength` is best swept over decades (1e-4 … 1e-1). Pick the
"elbow": smallest value that removes noise/ringing without softening structure.
