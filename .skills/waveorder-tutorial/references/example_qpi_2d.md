# Example config: 2D phase from the QPI-defocus sample

The Stage 3 draft config for the sample that `scripts/get_example_data.py`
downloads, already in the draft form Stage 3 asks for: autofocus on (an
optimizable `z_focus_offset`) and a small `regularization_strength`. As with
any labelfree draft, save the block below twice in `1-draft-recon/`, as
`invert_false.yml` and as `invert_true.yml` with `invert_phase_contrast: true`,
and reconstruct both.

```yaml
input_channel_names:                        # matches the 'BF' channel in the sample
- BF
time_indices: all
reconstruction_dimension: 2                 # 2D phase from a defocus stack
phase:
  transfer_function:
    wavelength_illumination: 0.532          # µm
    yx_pixel_size: 0.325                    # µm (6.5 µm camera / 20x)
    z_pixel_size: 2.0                       # µm (defocus step)
    z_padding: 0
    z_focus_offset:                         # autofocus for a 2D draft
      init: 0
      lr: 0.1
    index_of_refraction_media: 1.0          # imaging in air
    numerical_aperture_detection: 0.55
    numerical_aperture_illumination: 0.4
    tilt_angle_zenith: 0.0                  # no shadow-cast contrast in this data
    tilt_angle_azimuth: 0.0
    invert_phase_contrast: false            # second copy: true; pick the correct density sign
  apply_inverse:
    reconstruction_algorithm: Tikhonov
    regularization_strength: 0.001        # start small; raise last, after the forward model is tuned
    TV_rho_strength: 0.001
    TV_iterations: 1
```
