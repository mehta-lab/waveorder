---
name: visualize-otf-from-config
description: Compute a waveorder transfer function (OTF) from a reconstruction config and open it in napari. Use when someone wants to inspect/visualize the transfer function or OTF for a config, check how NA / wavelength / pixel-size choices shape the transfer function, or debug a config before reconstructing. Runs `wo compute-tf` then `wo view`.
---

# Visualize the OTF from a config

Compute the transfer function for a `waveorder` reconstruction config and open it
in napari. Exactly two CLI calls.

## Inputs

- **config** — a reconstruction config YAML (the same kind used by `wo rec`; e.g.
  a `phase:` or `fluorescence:` config). This defines the optics (NA, wavelength,
  pixel sizes, RI, tilt, …) that shape the transfer function.
- **input position** — a path to one OME-Zarr position (`.../row/col/fov`).
  `wo compute-tf` needs it only to read the **array shape**; the values are not
  used, so any dataset with the right ZYX shape works.

If the user gave a `/visualize-otf-from-config <config> <input>` argument, use it.
Otherwise ask for the config path and the input position path.

## The two calls

```bash
# 1. Compute the transfer function
wo compute-tf -i <input.zarr>/0/0/0 -c <config.yml> -o ./otf.zarr

# 2. Open it in napari (backgrounded so it doesn't block)
wo view ./otf.zarr &
```

Notes:
- The subcommand is `compute-tf` (alias for `compute-transfer-function`) — there
  is no `calc-tf`.
- `wo view` auto-detects a transfer function (it has `settings` in its zarr
  attrs) and shows each component's real/imag parts with a diverging `bwr`
  colormap, `ifftshift`-ed so DC is centered.
- `wo view` starts napari's event loop and **blocks the terminal until closed**.
  When you run it, launch it in the background (`&`, or the Bash tool's
  `run_in_background: true`), tell the user the window is open, and let them
  inspect it — do not render a static PNG.
- If `wo` isn't on PATH, prefix both calls with `uv run` from the repo.

That's it — report the `./otf.zarr` path and that napari is open.
