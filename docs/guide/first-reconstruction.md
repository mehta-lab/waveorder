This guide walks you through your first `waveorder` reconstruction, from raw
imaging data to a tuned result you trust. It is organized as six stages:
decide if waveorder fits your use case, identify your data and metadata,
run draft reconstructions, refine them manually, refine them automatically,
and scale up to your full dataset.

Prefer a guided session? This tutorial is paired with an agentic skill: open an
agent that supports skills (for example [Claude Code](https://claude.com/claude-code))
in the waveorder repository and run `/waveorder-tutorial`. The agent will walk
you through the same stages interactively, run the commands, and help you
interpret the results in napari.

## Stage 1: Is waveorder right for you?

waveorder performs physics-informed reconstructions of labelfree and
fluorescence microscopy data, which can sharpen, denoise, improve contrast,
and improve interpretability for downstream tasks. Good results require you to
be reasonably familiar with your data: what is the sample, how is the data
stored, and how was the microscope prepared during acquisition? Small metadata
gaps can be reliably filled by guessing or estimating from the data, but
complete absence of metadata will lead to poor results.

waveorder works best on volumetric labelfree and fluorescence microscopy data
acquired from weakly scattering samples. This typically means relatively thin
and transparent biological specimens, but in our experience waveorder can be
enabling for samples as thick and absorbing as zebrafish embryos.

waveorder works best with a small amount of manual tuning during initial
setup, followed by automatic tuning as it is deployed to larger datasets. We
recommend setting aside a couple of hours to manually tune your first
reconstructions.

If that matches your use case, install the CLI with napari and confirm it
works:

```bash
pip install "waveorder[all]"
wo --help
```

If you have no data of your own to practice on, the tutorial skill can fetch a
small brightfield defocus stack (the QPI-from-defocus demo dataset) that runs
through the whole workflow in minutes.

## Stage 2: What data and metadata do you have?

### Dimensions of your input dataset

waveorder takes single- and multi-channel volumetric data, (C)ZYX, as input
and outputs single- and multi-channel volumes or slices, (C)(Z)YX.

- **YX data:** waveorder does not currently support 2D YX inputs; a defocus
  stack is required.
- **ZYX data:** waveorder supports several labelfree and fluorescence
  reconstructions.
- **CZYX data:** if the C channels are independent, reconstruct them
  separately as individual labelfree or fluorescence reconstructions. For
  example, a brightfield channel and two fluorescence channels become three
  separate ZYX reconstructions. If some channels are dependent and they image
  an object that does not move much, waveorder supports polarization- and
  illumination-angle-diverse reconstructions.
- **Time and position dimensions:** fully supported; every point along these
  dimensions is reconstructed independently.

### Minimum size of a representative input

Choose a minimal dataset so that you can iterate quickly on your initial
reconstructions. Ask yourself: what are the smallest values of (C), Z, Y, and
X that contain a biological unit of interest? For example, if you are imaging
and analyzing single cells that fit within Z = 10, Y = 512, X = 512 voxels,
then 10 x 512 x 512 is a representative input.

If the representative size is less than half a single field of view, crop an
ROI around a single biological unit of interest (open the data in napari,
hover over the unit, and read the coordinates from the status bar). If it is
larger, use a single field of view. If your ROI would fill more than about 20%
of your available RAM, expect slower iteration; consider more RAM or a smaller
ROI.

### Contrast type of each channel

waveorder needs each channel's name along with its contrast type: classify
each channel as **fluorescence**, **labelfree**, or **polarization**.

### Voxel size

waveorder needs the size of each voxel in X, Y, and Z, in micrometers. Start
from the dataset's metadata. If the metadata says 1.0 or is empty, calculate
the YX pixel size from the camera pixel size divided by the total
magnification (for example, a 6.5 µm camera pixel behind a 20x objective gives
0.325 µm), and use the Z step you set during acquisition.

### Imaging conditions

waveorder needs the NA of illumination, the NA of detection, the wavelength,
and the index of refraction of the immersion media.

- For NAs, use what is stamped on the barrel of the objective (and condenser).
  Fluorescence needs only the detection NA.
- Labelfree wavelength is the illumination wavelength; fluorescence wavelength
  is the emission wavelength. If you don't know or you used a broadband
  source, 0.5 µm is a reasonable guess.
- Index of refraction: 1.0 for air, about 1.33 for water, about 1.47 for oil.

All of these parameters are reasonably forgiving, and the tuning stages below
will catch meaningful mismatches. For fluorescence, also note the modality:
widefield, confocal (record the pinhole diameter), iSIM, or light sheet.

### Reconstruction dimension: 2D or 3D?

waveorder can output 3D and 2D reconstructions, useful for thick and thin
samples relative to the depth of field of the microscope. Does your entire
sample go in and out of focus all at the same time? If yes, your sample is
thin compared to the depth of field: use a 2D reconstruction. If no, use a 3D
reconstruction. Trialling both doesn't hurt, so in an edge case inspect both.

## Stage 3: Draft reconstructions

Convert your data into OME-Zarr. The `iohub` package (installed with
waveorder) provides reliable conversion from Micro-Manager datasets via
`iohub convert -i <dataset> -o data.zarr`; the tutorial skill includes scripts
for bare TIFFs, numpy arrays, and plain zarr arrays. Organize your work into
subfolders like `0-conversion`, `1-draft-recon`, `2-manual-sweep`, and
`3-automated-sweeps`.

Write a small set of draft configuration files from your Stage 2 answers
(templates live in `docs/examples/cli/configs/`):

- For labelfree reconstructions, start with two configs:
  `invert_phase_contrast: true` and `false`.
- If you are unsure about 2D vs 3D, double the number of configs with
  `reconstruction_dimension: 2` and `3`.
- For 2D reconstructions, start with autofocus by making the focus offset
  optimizable: `z_focus_offset: {init: 0, lr: 0.1}`.

Run these 2-4 draft reconstructions and open them in napari alongside the raw
data:

```bash
wo rec -i 0-conversion/data.zarr/*/*/* -c 1-draft-recon/config.yml -o 1-draft-recon/recon.zarr
wo view 0-conversion/data.zarr 1-draft-recon/*.zarr
```

Decide `invert_phase_contrast` by choosing the reconstruction that sets
objects denser than the surrounding media brighter than background: nucleoli
and membranes tend to be denser than background, so they should appear bright.
Decide 2D vs 3D by preference after inspecting both.

## Stage 4: Manually refined reconstructions

View the reconstructions alongside the raw data in napari and describe any
problems you see. Common problems include:

- too noisy (noisier than the raw data)
- too smooth (smoother than the raw data)
- top-to-bottom wrapping (top and bottom slices are similar)
- ringing (ripples near edge structures)
- tilted point spread functions (a small structure appears to move sideways as
  you move through focus)
- "shadow-cast" labelfree contrast (in-focus structures look bright on one
  side and dark on the other)

Based on what you see, run manually chosen parameter sweeps. Aim for about
five reconstructions per sweep, viewed side by side. If your favorite is at
the edge of the sweep, redo the sweep centered on it; if the step between
neighboring reconstructions is too large, run a refined sweep.

Suggestions for each symptom:

- Too noisy or too smooth: sweep `regularization_strength` over decades.
- 2D reconstruction slightly defocused: sweep `z_focus_offset` to check
  whether autofocus is failing.
- Top-to-bottom wrapping: increase `z_padding`.
- Ringing in labelfree reconstructions: sweep
  `numerical_aperture_illumination`.
- Tilted point spread function: sweep the tilt parameters together
  (`tilt_angle_azimuth` and `tilt_angle_zenith`) and view the sweep on two
  napari sliders.
- Shadow-cast contrast: the tilt is likely quite large, requiring several
  iterations of `tilt_angle_azimuth` and `tilt_angle_zenith` optimization.
- Sweep any uncertain parameters to search for model mismatch, and note which
  sweeps result in improvements.

Keep sweeping until you reach a reconstruction that is an improvement compared
to your raw data, and iterate until you are satisfied and see no further
consistent improvements.

## Stage 5: Automatically refined reconstructions

Trial your manually tuned configuration on other time points, positions, or
other ROIs within the field of view. If the reconstructions look good, skip
automatic refinement. If some fail, inspect the failures as in Stage 4 to
identify which parameter needs to be re-tuned; those parameters are your
auto-tuning candidates.

waveorder can optimize `z_focus_offset`, `tilt_angle_azimuth`,
`tilt_angle_zenith`, `numerical_aperture_illumination`, and
`numerical_aperture_detection`. Mark a parameter optimizable with an
`init`/`lr` pair and start with 10 iterations and a learning rate of 0.1:

```yaml
phase:
  transfer_function:
    z_focus_offset:
      init: 0
      lr: 0.1
optimization:
  max_iterations: 10
  method: adam
```

Trial these auto-tuning approaches one at a time and check the results against
your manually chosen values. Can the values you found by eye be found
automatically? If yes, keep those parameters optimizable in your production
configs; if not, fix them at your manual values.

## Stage 6: Scale the reconstruction

Prepare an estimate of the resources required to complete the entire
reconstruction: wall time (time one volume, multiply by time points x
positions x channels), memory, and storage. If these resources are available
and acceptable, proceed with the complete reconstruction.

If not, onboard some combination of:

- **Parallelization** (reduces wall time): run one `wo rec` per position or
  time chunk, locally or as cluster jobs.
- **Tiling** (reduces memory and wall time): reconstruct per tile and stitch
  with `wo tile-stitch`; the tile-stitch algorithms in
  [biahub](https://github.com/czbiohub-sf/biahub) are a good starting point
  for parallelizing across tiles.
- **GPU usage** (typically reduces wall time): set `device: auto` or
  `device: cuda:0` in the config.
- **Deleting intermediate zarr stores** (reduces storage): converted data and
  transfer functions can be regenerated from the config.

You now have a tuned, validated configuration and a plan for the full dataset.
For deeper reference material, see the
[reconstruction guide](./reconstruction-guide.md) and the example configs in
`docs/examples/`.
