---
name: waveorder-visualize-otf
description: Compute a waveorder transfer function (OTF) from a reconstruction config on the user's own computer and open it in napari. Use when someone wants to inspect/visualize the transfer function or OTF for a config, check how NA / wavelength / pixel-size choices shape the transfer function, or debug a config before reconstructing. Runs `wo compute-tf` then `wo view`, so it needs a client that can run shell commands locally, with uv and a display. Not for reconstructing data, which waveorder-tutorial covers.
---

# Visualize the OTF from a config

Compute the transfer function for a `waveorder` reconstruction config and open it
in napari. Exactly two CLI calls, run on the user's own computer; a client that
cannot run shell commands cannot follow this skill.

## Inputs

- **config**: a reconstruction config YAML (the same kind used by `wo rec`; e.g.
  a `phase:` or `fluorescence:` config). This defines the optics (NA, wavelength,
  pixel sizes, RI, tilt, ...) that shape the transfer function.
- **input position**: a path to one OME-Zarr position (`.../row/col/fov`).
  `wo compute-tf` needs it only to read the **array shape**; the values are not
  used, so any dataset with the right ZYX shape works.

If the user gave a `/waveorder-visualize-otf <config> <input>` argument, use it.
Otherwise ask for the config path and the input position path.

## The two calls

Name the output after the config, so comparing two settings never overwrites
an earlier result (`phase_3d.yml` becomes `./phase_3d_otf.zarr`):

```bash
# 1. Compute the transfer function
wo compute-tf -i <input.zarr>/0/0/0 -c <config>.yml -o ./<config>_otf.zarr

# 2. Open it in napari (backgrounded so it doesn't block)
wo view ./<config>_otf.zarr </dev/null &
```

To compare settings, run step 1 once per config and open all results in one
`wo view` call.

Notes:
- The subcommand is `compute-tf` (alias for `compute-transfer-function`); there
  is no `calc-tf`.
- `wo view` auto-detects a transfer function (it has `settings` in its zarr
  attrs) and shows real/imag parts with a diverging `bwr` colormap, `ifftshift`-ed
  so DC is centered. **3D** reconstructions store the transfer function directly.
  **2D** reconstructions store a singular system (`U`, `S`, `Vh`); `wo view`
  reconstructs the transfer function from the SVD (`H = U @ diag(S) @ Vh`) and
  shows that (e.g. the absorption and phase transfer functions for 2D phase)
  rather than the raw singular vectors.
- `wo view` starts napari's event loop and **blocks the terminal until closed**.
  Start it with your environment's long-running process facility; it must not
  block further commands or inherit interactive stdin (in a plain POSIX shell:
  `wo view ./<config>_otf.zarr </dev/null &`). Confirm the process does not exit
  immediately, tell the user the window is open, and let them inspect it. Do
  not render a static PNG.
- If `wo` isn't installed, run both calls without installing via
  `uvx --from "waveorder[all]" wo ...`.

That's it: report the `./<config>_otf.zarr` path and that napari is open.
