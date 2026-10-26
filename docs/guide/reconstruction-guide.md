# Automating reconstructions

`waveorder` uses a configuration-file-based command-line interface (CLI) to perform all reconstructions.

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

## Tiled reconstruction

`waveorder tile-stitch` reconstructs a position in overlapping tiles and blends them back into one volume, so each tile's transfer function is the size of a tile rather than the whole field. Its config has `tile` and `blend` sections, and the usual reconstruction settings go under `recon`. [`tile_stitch_phase_3d.sh`](../examples/cli/tile_stitch_phase_3d.sh) simulates a phase dataset and reconstructs it with [`configs/tile-stitch_phase_3d.yml`](../examples/cli/configs/tile-stitch_phase_3d.yml).

A 3D reconstruction has to list `z` in `tile.tile_size`, because the transfer function takes its shape from `tile_size`. Set it to the full depth to tile only in Y and X. `tile-stitch` reconstructs one timepoint per run, and `--device` overrides `recon.device`. To spread tiles across workers, see [biahub](https://github.com/czbiohub-sf/biahub).

## Input options

The input `-i` flag always accepts a list of inputs, either explicitly e.g. `-i ./data.zarr/A/1/0 ./data.zarr/A/2/0` or through wildcards `-i ./data.zarr/*/*/*`. The positions in a high-content screening `.zarr` store are organized into `/row/col/fov` folders, so `./input.zarr/*/*/*` creates a list of all positions in a dataset.

The `waveorder compute-tf` command accepts a list of inputs, but it only computes the transfer function for the first position in the list. The `apply-inv-tf` command accepts a list of inputs and applies the same transfer function to all of the inputs, which requires that all positions contain arrays with matching TCZYX shapes.

## What types of reconstructions are supported?
See `/waveorder/examples/` for a list of example configuration files.

WIP: This documentation will be expanded for each reconstruction type and parameter.
