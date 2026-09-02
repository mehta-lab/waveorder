---
name: waveorder-tutorial
description: Interactively walk a user through their first waveorder reconstruction, from raw imaging data to a tuned result. Use when someone has microscopy data (TIFF, zarr, or array) and wants to reconstruct phase (brightfield) or deconvolved fluorescence, or asks to "try waveorder", "reconstruct my data", or "set up a waveorder config". Guides suitability, install, data and metadata identification, OME-Zarr conversion, contrast and dimension choice, draft reconstructions, manual parameter sweeps, automatic tuning, and scaling up.
---

# waveorder first-reconstruction tutorial

Guide the user, stage by stage, from raw imaging data to a tuned `waveorder`
reconstruction they trust. This is an **interactive** skill: ask the questions,
wait for answers, run the commands, and interpret the results together. Do not
dump every question at once. Go one stage at a time and adapt to what the user
reports. A first reconstruction typically takes a couple of hours of manual
tuning; say so up front and pace the session accordingly.

The paired human-readable version of this workflow lives at
`docs/guide/first-reconstruction.md`. Keep your guidance consistent with it.

Supporting material in this skill:
- `references/conversion.md` - turning TIFF / zarr / array into OME-Zarr
- `references/config_reference.md` - every config knob, sweeps, and auto-tuning
- `references/example_qpi_2d.yml` - ready config for the QPI-defocus sample
- `scripts/get_example_data.py` - fetch the QPI-defocus sample OME-Zarr
- `scripts/to_omezarr.py` - converter (TIFF/npy/array to viewable OME-Zarr)
- `scripts/crop_roi.py` - crop an OME-Zarr position to a small ROI
- `scripts/sweep.py` - 1D and 2D parameter sweeps with side-by-side viewing

## The `wo` command

The CLI entrypoint is `wo` (alias for `waveorder`). Key subcommands:
- `wo rec -i <input.zarr>/*/*/* -c <config.yml> -o <output.zarr>` - reconstruct
- `wo view <zarr> [<zarr> ...]` - open datasets / transfer functions in napari
- `wo compute-tf -i <pos> -c <config.yml> -o <tf.zarr>` - transfer function only
- `wo sim -c <config.yml> -o <sim.zarr>` - simulate phantom + measurement
- `wo tile-stitch` (`wo ts`) - stitch multi-position mosaics

Input paths use the glob `input.zarr/*/*/*` to expand HCS plate positions.
If plain `wo` isn't on PATH, prefix commands with `uv run` from the repo.

## Viewing convention: always napari, never PNGs

**Always show data and reconstructions in napari via `wo view`. Never render
static PNGs / matplotlib figures / screenshots to inspect or present results.**
napari is interactive: the *user* scrolls Z, adjusts contrast, and toggles
layers, which is exactly what's needed to judge focus, density sign, noise, and
ringing. A flat PNG can't support that and would defeat the point.

`wo view` starts napari's event loop and **blocks the terminal until the window
is closed**. So when *you* (the assistant) launch it, run it in the background
so the session isn't stuck:

```bash
wo view ./data.zarr ./recon.zarr &      # background; napari opens for the user
```

(With the Bash tool, use `run_in_background: true`.) Then tell the user the
napari window is open, describe what to look for, and **ask them what they
observe**. Do not try to see the result yourself. Reconstructions come back
squeezed so 2D results (single Z) display correctly, and grid view is enabled
so input and output sit side by side.

## Organize the work

Create a working folder for the session and keep stages separated:

```
waveorder-tutorial/
├── 0-conversion/       # converted OME-Zarr + ROI crops
├── 1-draft-recon/      # draft configs + reconstructions
├── 2-manual-sweep/     # manual parameter sweeps
└── 3-automated-sweeps/ # auto-tuning trials
```

---

## Stage 1: Is waveorder right for you?

Start by giving the user this context, briefly and in your own words:

- waveorder performs physics-informed reconstructions of labelfree and
  fluorescence microscopy data, which can sharpen, denoise, improve contrast,
  and improve interpretability for downstream tasks. Good results require the
  user to be reasonably familiar with their data: what is the sample, how is
  the data stored, and how was the microscope prepared during acquisition.
  Small metadata gaps can be filled by guessing or estimating from the data,
  but complete absence of metadata will lead to poor results.
- waveorder works best on volumetric labelfree and fluorescence data from
  weakly scattering samples. This typically means relatively thin and
  transparent biological specimens, but waveorder can be enabling for samples
  as thick and absorbing as zebrafish embryos.
- waveorder works best with a small amount of manual tuning during initial
  setup, followed by automatic tuning as it is deployed to larger datasets.
  Recommend setting aside a couple of hours to manually tune the first
  reconstructions.

**Ask:** "Does this match your use case? Ready to install?"

If yes, install the CLI with napari and confirm it works:

```bash
wo --help                                    # is the CLI installed?
python -c "import napari; print(napari.__version__)"
```

- If `wo` is missing: `uv pip install "waveorder[all]"` (the `all` extra pulls
  in `napari[pyqt6]` and `napari-ome-zarr`). Developing the repo:
  `uv pip install -e ".[all]"` from the checkout.
- If napari imports but no Qt backend is present:
  `uv pip install "napari[pyqt6]"`.
- Confirm `wo --help` lists `reconstruct`, `view`, `simulate` before moving on.
- **No data of their own?** Offer the QPI-from-defocus sample:
  `python scripts/get_example_data.py` downloads a small OME-Zarr (a single
  `BF` brightfield defocus stack). Its Stage 2 answers are pre-filled in
  `references/example_qpi_2d.yml` (channel `BF`, labelfree, yx 0.325 µm,
  z 2.0 µm, n 1.0, wavelength 0.532 µm, NA det 0.55, NA ill 0.4, 2D), so with
  the sample you can move quickly through Stage 2 confirming each value.

---

## Stage 2: What data and metadata do you have?

Work through these one at a time. Record every answer; they all feed the
config in Stage 3.

### 2a. Dimensions of the input dataset

waveorder takes single- and multi-channel volumetric data, (C)ZYX, as input
and outputs single- and multi-channel volumes or slices, (C)(Z)YX.

**Ask:** "Give me a path to your dataset and describe its dimensions."

Inspect the data yourself too (file listing, metadata, or a quick array-shape
check) and reconcile with their description. Then branch:

- **YX only:** waveorder does not currently support 2D YX inputs. A defocus
  stack (multiple Z slices) is required. Stop here or point them to data
  with a Z dimension.
- **ZYX:** great, waveorder supports several labelfree and fluorescence
  reconstructions.
- **CZYX:** great. If the C channels are **independent** (for example one
  brightfield channel and two fluorescence channels), reconstruct them
  separately as individual ZYX reconstructions. If some channels are
  **dependent** (they jointly encode one object that does not move much
  between channels), waveorder supports polarization- and
  illumination-angle-diverse reconstructions; for a first tutorial steer to
  phase or fluorescence and mention that birefringence exists.
- **T and/or position dimensions:** fully supported. Every time point and
  position is reconstructed independently. For the tutorial, pick one time
  point and one position.

### 2b. Minimum representative input

We want a minimal dataset so we can iterate quickly on initial
reconstructions.

**Ask:** "What are the smallest (C), Z, Y, X that contain one biological unit
of interest? For example, if you are imaging single cells that fit within
Z = 10, Y = 512, X = 512 voxels, then 10 x 512 x 512 is representative."

- If the representative size is **less than half a single FOV**: walk the user
  through ROI selection in napari. Open the data (`wo view <path> &`), ask
  them to hover the cursor over one biological unit of interest and read the
  coordinates from napari's status bar (bottom left), then crop with
  `scripts/crop_roi.py` (see `references/conversion.md`). Iterate until they
  confirm the crop contains the unit.
- If the representative size is **more than half a FOV**: skip ROI selection
  and use a single FOV.
- Estimate the ROI's memory footprint (voxels x 4 bytes, times a few for
  intermediates) against available RAM (`free -g`). If the ROI would fill
  more than ~20% of available RAM, continue but warn: iteration might be
  slow; suggest more RAM, a smaller ROI, or patience.

### 2c. Contrast type of each channel

waveorder needs each channel's name and contrast type.

**Ask:** For each channel, offer your best guess from the channel name
(for example "BF" or "Phase3D" is labelfree; "GFP"/"mCherry"/dye names are
fluorescence; "State0..3" is polarization) and have the user confirm:
**fluorescence**, **labelfree**, or **polarization**.

### 2d. Voxel size

waveorder needs the size of each voxel in X, Y, and Z, in micrometers.

**Ask:** Read the scale from the input metadata and present it as a starting
point; ask the user to confirm. If the metadata says 1.0, is empty, or the
user does not know, walk them through the calculation:

- yx pixel size = camera pixel size / total magnification
  (for example 6.5 µm camera pixel / 20x = 0.325 µm).
- z step = the axial spacing the user set during acquisition.

Sanity check: a 63x/1.4 objective on a typical camera gives roughly 0.1 µm yx.

### 2e. Imaging conditions

waveorder needs the NA of illumination, NA of detection, wavelength, and index
of refraction of the immersion media.

**Ask:** Using the dataset metadata as a starting guess, confirm per channel
group:

- **NA detection:** suggest what is stamped on the objective barrel.
- **NA illumination (labelfree only):** the condenser NA. Fluorescence needs
  no NA illumination.
- **Wavelength:** labelfree uses the illumination wavelength; fluorescence
  uses the emission wavelength. If unknown or broadband, suggest 0.5 µm.
- **Index of refraction of the immersion media:** 1.0 air, ~1.33 water,
  ~1.47 oil.

These parameters are reasonably forgiving; best guesses are fine to start,
and Stages 4 and 5 will catch meaningful mismatches.

**Ask (fluorescence only):** "Widefield, confocal, iSIM, or light sheet?"
- Widefield: `confocal_pinhole_diameter: null`.
- Confocal: set `confocal_pinhole_diameter`.
- iSIM / light sheet: not modeled explicitly; the widefield model with the
  effective NA is a workable approximation. Say so honestly.

### 2f. Reconstruction dimension: 2D or 3D?

waveorder outputs 3D volumes or 2D slices, suited to thick and thin samples
(relative to the microscope's depth of field).

**Ask:** "Does your entire sample go in and out of focus all at the same time?
In other words, is your sample thin compared to the depth of field?"

- Yes (thin): suggest a **2D** reconstruction.
- No (thick): suggest a **3D** reconstruction.
- Unsure / edge case: trialling both doesn't hurt; Stage 3 runs both and the
  user picks by inspection.

---

## Stage 3: Convert and draft reconstructions

### Convert to OME-Zarr

waveorder reads OME-Zarr (HCS layout, TCZYX, named channels, pixel-size
scale). Convert into `0-conversion/`:

- **Already OME-Zarr:** no conversion; confirm with `wo view`.
- **Micro-Manager TIFF dataset:** the `iohub` package (installed with
  waveorder) provides reliable CLI conversion:
  `iohub convert -i <mm_tiff_dir> -o 0-conversion/data.zarr`.
- **Bare TIFF / npy / plain zarr:** `scripts/to_omezarr.py` with `--axes`,
  `--channel-name`, and the Stage 2d pixel sizes.
- **ROI crop (from 2b):** `scripts/crop_roi.py` on the converted store.

Details and flags: `references/conversion.md`. Always verify with
`wo view 0-conversion/data.zarr &`: correct channel names, and scrolling Z
moves through focus.

### Draft configs

Write configs into `1-draft-recon/` from the Stage 2 answers (templates:
`docs/examples/cli/configs/{phase,fluorescence}_{2d,3d}.yml`; every knob:
`references/config_reference.md`).

- **Labelfree:** start with **two** configs, `invert_phase_contrast: true`
  and `false`. This knob is hard to set from theory; we decide it by looking.
- **Unsure about 2D vs 3D:** double the set with
  `reconstruction_dimension: 2` and `3` (four configs).
- **2D reconstructions:** start with autofocus on, i.e. make the focus offset
  optimizable:

  ```yaml
  z_focus_offset:
    init: 0
    lr: 0.1
  ```

  (There is no literal `auto`; the `init`/`lr` form plus the default
  `optimization:` behavior is waveorder's autofocus. If optimization is slow
  or misbehaves, fall back to a plain `z_focus_offset: 0` and sweep it in
  Stage 4.)
- Keep `tilt_angle_zenith` and `tilt_angle_azimuth` at 0.0 for drafts.
- **Fluorescence:** start with `Tikhonov`. Mention that `RL` and `RLGC`
  (Richardson-Lucy) exist for 3D fluorescence once the basics work.

### Run and decide

```bash
wo rec -i 0-conversion/data.zarr/*/*/* -c 1-draft-recon/<cfg>.yml -o 1-draft-recon/<name>.zarr
wo view 0-conversion/data.zarr 1-draft-recon/*.zarr &
```

Run the 2-4 drafts, open them together with the raw data, and decide with the
user:

- **`invert_phase_contrast`:** ask the user to choose the reconstruction in
  which objects denser than the surrounding media appear brighter than
  background. Prompt: nucleoli and membranes tend to be denser than
  background, so they should appear bright. (Nuclei overall, vacuoles, and
  lumen are less dense and should appear dark.)
- **2D vs 3D:** the user's preference after inspecting both.

Carry the winning config into Stage 4.

---

## Stage 4: Manually refined reconstructions

Show the reconstruction alongside the raw data in napari and **ask the user to
describe what they see and any problems**. Common problems to name for them:

- **too noisy** (noisier than the raw data)
- **too smooth** (smoother than the raw data)
- **top-to-bottom wrapping** (top and bottom Z slices look similar)
- **ringing** (ripples near edge structures)
- **tilted point spread function** (a small structure appears to move
  sideways as you scroll through focus)
- **"shadow-cast" labelfree contrast** (in-focus structures look bright on
  one side and dark on the other, like DIC)

Based on their feedback, run manually chosen sweeps with `scripts/sweep.py`.
Aim for **about 5 reconstructions per sweep**, opened side by side:

```bash
python scripts/sweep.py -i 0-conversion/data.zarr -c best.yml \
    --param regularization_strength --values 1e-4 1e-3 1e-2 1e-1 1e0 \
    --outdir 2-manual-sweep
```

Sweep rules of thumb:

- If their favorite is at the **edge** of the sweep, redo the sweep centered
  on it.
- If the step between neighbors is too coarse, run a refined sweep between
  the two best values.

Symptom-to-sweep map:

- **Too noisy or too smooth:** sweep `regularization_strength` over decades.
- **2D reconstruction looks defocused:** sweep `z_focus_offset` to check
  whether autofocus failed.
- **Top-to-bottom wrapping:** increase `z_padding` (try ~a quarter to half
  the Z stack).
- **Ringing (labelfree):** sweep `numerical_aperture_illumination`; if
  ringing persists at a good noise level, try `reconstruction_algorithm: TV`.
- **Tilted point spread function:** sweep both tilt parameters at once and
  show the user the 2D sweep on two napari sliders:

  ```bash
  python scripts/sweep.py -i 0-conversion/data.zarr -c best.yml \
      --param tilt_angle_azimuth --values 0 1.57 3.14 4.71 \
      --param2 tilt_angle_zenith --values2 0.05 0.1 0.2 \
      --outdir 2-manual-sweep
  ```

- **Shadow-cast contrast:** the tilt is likely large; expect several
  iterations of `tilt_angle_azimuth` / `tilt_angle_zenith` refinement
  (coarse azimuth first, then zenith, then refine).
- **Uncertain metadata:** sweep any parameter the user was unsure about in
  Stage 2 to search for model mismatch. Which sweeps produce improvements?

Keep sweeping until the user reaches a reconstruction that is an improvement
over the raw data, and iterate until they are satisfied and see no further
consistent improvements. Save the winning config.

---

## Stage 5: Automatically refined reconstructions

First test generalization: run the manually tuned config on **other time
points, positions, or other ROIs within the FOV** and view the results with
the user.

- If they look good everywhere: **skip auto-tuning** and go to Stage 6.
- If some fail: walk the user through the Stage 4 inspection on a failed case
  to identify which parameter drifts (focus offset and tilt are the usual
  suspects). Those parameters are the auto-tuning candidates.

With candidate parameters in hand, trial waveorder's auto-tuning **one
parameter at a time** in `3-automated-sweeps/`. Mark a parameter optimizable
with an `init`/`lr` pair and add an `optimization:` block; start with 10
iterations and learning rate 0.1:

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

Optimizable parameters: `z_focus_offset`, `tilt_angle_azimuth`,
`tilt_angle_zenith`, `numerical_aperture_illumination`,
`numerical_aperture_detection`. `wo rec` runs the optimization on the first
position, writes `<config>_optimized.yml` with the fitted values, then
reconstructs. See `references/config_reference.md` for methods and losses.

**Check each trial against the user's manually chosen values**: does
auto-tuning find the value the user picked by eye (on the tuned FOV) and fix
the failed cases? If yes, keep that parameter optimizable in the production
config. If not, keep it fixed at the manual value.

---

## Stage 6: Scale the reconstruction

Prepare an estimate of the resources for the full dataset:

- **Wall time:** time one (C)ZYX reconstruction (Stage 4 winner at full FOV),
  multiply by time points x positions x channels.
- **Memory:** the full-FOV footprint measured in Stage 2b.
- **Storage:** output size per position x positions, plus the converted input
  and transfer-function stores.

Present the estimate. If the resources are available and acceptable, run the
complete reconstruction (`wo rec` accepts many positions in one call, and
`time_indices: all` covers time).

If not, work with the user to onboard some combination of:

- **Parallelization** (reduces wall time): launch one `wo rec` per position
  or time chunk, in parallel locally or as cluster jobs (one job per
  position; `time_indices` accepts a list for time chunking).
- **Tiling** (reduces memory and wall time): split large FOVs into tiles,
  reconstruct per tile, and stitch with `wo tile-stitch`. For parallelizing
  across tiles, use the tile-stitch algorithms in
  https://github.com/czbiohub-sf/biahub as inspiration.
- **GPU** (typically reduces wall time): set `device: auto` (or `cuda:0`,
  `mps`) in the config.
- **Deleting intermediate zarr stores** (reduces storage): the converted
  input (if the raw data is retained) and transfer-function stores can be
  regenerated from the config.

Wrap up by summarizing the final config, where each parameter came from
(metadata, user knowledge, manual sweep, or auto-tuning), and the resource
plan. Save everything in the staged folders.
