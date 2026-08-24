---
name: waveorder-tutorial
description: Interactively walk a user through their first waveorder reconstruction from a raw imaging file via the CLI. Use when someone has a microscopy image (TIFF, zarr, or array) and wants to reconstruct phase (brightfield) or deconvolved fluorescence, or asks to "try waveorder", "reconstruct my data", or "set up a waveorder config". Guides environment setup, OME-Zarr conversion, config authoring, viewing in napari, and parameter sweeps.
---

# waveorder reconstruction tutorial

Guide the user, step by step, from a raw imaging file to a first `waveorder`
reconstruction they can inspect in napari. This is an **interactive** skill:
ask the questions, wait for answers, run the commands, and interpret the
results together. Do not dump every question at once — go one stage at a time
and adapt to what the user reports.

Supporting material in this skill:
- `references/conversion.md` — how to turn TIFF / zarr / array into OME-Zarr
- `references/config_reference.md` — every config knob explained
- `references/example_qpi_2d.yml` — ready config for the QPI-defocus sample
- `scripts/get_example_data.py` — fetch the QPI-defocus sample OME-Zarr
- `scripts/to_omezarr.py` — converter (TIFF/npy/array → viewable OME-Zarr)
- `scripts/sweep.py` — regularization / parameter sweep + side-by-side view

## The `wo` command

The CLI entrypoint is `wo` (alias for `waveorder`). Key subcommands:
- `wo rec -i <input.zarr>/*/*/* -c <config.yml> -o <output.zarr>` — reconstruct
- `wo view <zarr> [<zarr> ...]` — open datasets / transfer functions in napari
- `wo sim -c <config.yml> -o <sim.zarr>` — simulate phantom + measurement

Input paths use the glob `input.zarr/*/*/*` to expand HCS plate positions.
If plain `wo` isn't on PATH, prefix commands with `uv run` from the repo.

## Viewing convention — always napari, never PNGs

**Always show data and reconstructions in napari via `wo view`. Never render
static PNGs / matplotlib figures / screenshots to inspect or present results.**
napari is interactive — the *user* scrolls Z, adjusts contrast, and toggles
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
observe** — do not try to see the result yourself. Reconstructions come back
squeezed so 2D results (single Z) display correctly, and grid view is enabled so
input and output sit side by side.

---

## Step 0 — Check the environment

Verify the CLI and napari before anything else:

```bash
wo --help                                   # is the CLI installed?
wo view 2>&1 | head -1                       # will error, but confirms import
python -c "import napari; print(napari.__version__)"
```

- If `wo` is missing, install into the user's env with uv:
  `uv pip install "waveorder[all]"` (the `all`/`visual` extra pulls in
  `napari[pyqt6]` and `napari-ome-zarr`). If they're developing the repo:
  `uv pip install -e ".[all]"` from the waveorder checkout.
- If napari imports but no Qt backend is present, `uv pip install "napari[pyqt6]"`.
- Confirm `wo --help` lists `reconstruct`, `view`, `simulate` before moving on.
- If the user has no data of their own to practice on, note that
  `scripts/get_example_data.py` can fetch a zenodo sample OME-Zarr (Step 1).

---

## Step 1 — Get the data into a viewable OME-Zarr

Ask: **"What form is your data in?"**
1. **an OME-Zarr** (already in `waveorder`'s native format) — **preferred**
2. a TIFF / OME-TIFF (single file or a folder of tiles)
3. an in-memory / `.npy` array or a plain (non-OME) zarr

The key thing `waveorder` needs is a **TCZYX** OME-Zarr with named channels and
a correct pixel scale.

**No data of their own?** Offer the QPI-from-defocus sample (the dataset from
`docs/examples/demos/QPI_defocus`). It's already an OME-Zarr — a single `BF`
brightfield defocus stack — so it skips conversion and is set up for a **2D
phase-from-defocus** reconstruction:
```bash
python scripts/get_example_data.py            # downloads + prints paths & commands
```
Use the printed `raw_data.zarr` path in place of `./data.zarr` below. This
example's answers are: contrast = brightfield→phase, dimension = **2D**, and the
optical parameters are pre-filled in `references/example_qpi_2d.yml`
(BF channel, yx 0.325 µm, z 2.0 µm, n 1.0, λ 0.532 µm, NA_det 0.55, NA_ill 0.4).

**Case 1 — OME-Zarr:** no conversion. Ask the user to paste the path, then just
confirm the axis order and channel names:
`wo view <path>` — the stack should focus/defocus as you scroll Z.

**Case 2 — TIFF:** ask the user to **paste the path to their TIFF** (a single
file, or a folder of tiles). Then convert with `scripts/to_omezarr.py`:
```bash
python scripts/to_omezarr.py <TIFF_PATH> ./data.zarr \
    --channel-name Brightfield --axes ZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

**Case 3 — array / plain zarr:** save to `.npy` if needed, then use
`scripts/to_omezarr.py` with `--axes` describing the dimension order.

See `references/conversion.md` for per-case flags. Then always:
- Confirm the result visually: `wo view ./data.zarr` — check Z focus and that
  channel names look right.
- Note the array shape and pixel sizes; you'll reuse them in the config.

---

## Step 2 — Choose contrast type and dimension

Ask two questions:

**A. Contrast type?**
- **Brightfield → phase** (label-free density). Use the `phase:` config block.
- **Fluorescence → deconvolution** (denoise + sharpen). Use `fluorescence:`.
- (Polarization/birefringence is out of scope for a first tutorial; mention it
  exists but steer to phase or fluorescence.)

**B. Reconstruction dimension — 2D or 3D?**
- **2D** works best when the object is **thin compared to the depth of focus** —
  i.e., the whole object appears to go in and out of focus at the same time as
  you scroll Z. Output is a single in-focus plane.
- **3D** for thick samples where different depths focus at different Z. Output
  is a volume. Start here if unsure and the sample is clearly volumetric.

Record the answers → this picks the config template and
`reconstruction_dimension: 2` or `3`.

---

## Step 3 — Fill in the configuration

Copy the matching template from the waveorder repo
(`docs/examples/cli/configs/{phase,fluorescence}_{2d,3d}.yml`) or write one from
`references/config_reference.md`. Ask the user for each optical parameter and
fill it in. Always set `input_channel_names` to the exact channel name in
their OME-Zarr.

Ask for:
- **`yx_pixel_size`** (µm) and **`z_pixel_size`** (µm) — from the acquisition;
  reuse what Step 1 recorded.
- **`numerical_aperture_detection`** — the objective NA.
- **`index_of_refraction_media`** — 1.0 air, ~1.33 water, ~1.47 oil, etc.
- **wavelength** — `wavelength_illumination` (phase) or `wavelength_emission`
  (fluorescence), in µm.
- Phase only: **`numerical_aperture_illumination`** — the condenser NA (must be
  ≤ `index_of_refraction_media`).
- Fluorescence only: **`confocal_pinhole_diameter`** — `null` for widefield.

**Flag `invert_phase_contrast` explicitly (phase only).** This is the knob most
likely to be wrong on the first try, and it's hard to set from theory — it
depends on the microscope's contrast convention. Plan to try **both `true` and
`false`** and pick the one with the correct density sign:
- **less-dense regions** (nuclei, vacuoles, lumen) should appear **darker than
  background**
- **more-dense regions** (cytoplasm, membranes, dense organelles) should appear
  **brighter than background**

Start `tilt_angle_zenith` and `tilt_angle_azimuth` at **0.0** (see Step 4).

Save the config next to the data, e.g. `./config.yml`.

---

## Step 4 — Run it and inspect in napari

```bash
wo rec -i ./data.zarr/*/*/* -c ./config.yml -o ./recon.zarr
wo view ./data.zarr ./recon.zarr &      # napari, backgrounded (see convention above)
```

Open the input and reconstruction together in napari (never a PNG). Tell the
user the window is open, then review the result **with them** — ask what they
see. Look for:
- **Density sign (phase):** apply the nuclei-dark / cytoplasm-bright test above.
  If inverted, flip `invert_phase_contrast` and re-run. This comes first —
  everything else is easier to judge once the sign is right.
- **Noise / graininess:** speckly, high-frequency texture → regularization is
  too low. Go to Step 5.
- **Ringing / halos:** dark/bright overshoot around edges → regularization too
  low, or try the `TV` algorithm.
- **Over-smoothing / loss of detail:** regularization too high → Step 5.

**Tilt:** keep `tilt_angle_zenith = 0.0` unless the user can *explicitly see*
directional "gradient- or shadow-casting" contrast in the in-focus data — an
image that looks lit from one side, like DIC. Only then add a small zenith tilt
and set the azimuth to point along the shadow direction. Don't add tilt to
chase artifacts that aren't shadow-cast.

---

## Step 5 — Parameter sweeps

When the user reports noise, ringing, or blur, sweep regularization instead of
guessing. `scripts/sweep.py` writes one config per value, reconstructs each, and
opens all results side by side in napari (grid view) for comparison — again, no
PNGs:

```bash
python scripts/sweep.py -i ./data.zarr -c ./config.yml \
    --param regularization_strength --values 1e-4 1e-3 1e-2 1e-1
```

The script launches napari at the end, which blocks. When *you* run it, use
`run_in_background: true`; or pass `--no-view` and open the results yourself in a
backgrounded `wo view ...`. Then ask the user which value looks best.

Guidance:
- **regularization_strength** is the main dial. Sweep over decades
  (e.g. 1e-4 → 1e-1). Higher = smoother/less noise but more blur; lower =
  sharper but noisier with more ringing.
- If ringing persists at a good noise level, sweep
  `reconstruction_algorithm` between `Tikhonov` and `TV` (TV suppresses ringing
  while keeping edges, at higher compute cost; tune `TV_rho_strength` /
  `TV_iterations`).
- Pick the value at the "elbow" — the smallest regularization that removes the
  objectionable noise/ringing without visibly softening real structures.
- Other sweepable knobs for fine-tuning: `z_focus_offset`,
  `numerical_aperture_illumination`. (waveorder can also *optimize* some of
  these automatically — mention `wo rec` with an `lr` key / the `optimization:`
  block if the user wants that later.)

Wrap up by saving the final config and reconstruction, and summarizing the
chosen parameters for the user.
