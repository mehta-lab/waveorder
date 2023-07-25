# Automating reconstructions

`recOrder` uses a configuration-file-based command-line interface (CLI) to perform all reconstructions.

## How can I use `recOrder`'s CLI to perform reconstructions?
`recOrder`'s CLI is summarized in the following figure:
<img src="./images/cli_structure.png" align="center">

The main command `reconstruct` command is composed of two subcommands: `compute-tf` and `apply-inv-tf`. 

A reconstruction can be performed with a single `reconstruct` call. For example:
```
recorder reconstruct data.zarr/0/0/0 -c config.yml -o reconstruction.zarr
```
Equivalently, a reconstruction can be performed with a `compute-tf` call followed by an `apply-inv-tf` call. For example:
```
recorder compute-tf data.zarr/0/0/0 -c config.yml -o tf.zarr
recorder apply-inv-tf data.zarr/0/0/0 tf.zarr -c config.yml -o reconstruction.zarr
```
Computing the transfer function is typically the most expensive part of the reconstruction, so saving a transfer function then applying it to many datasets can save time. 

## What types of reconstructions are supported?
See `/recOrder/examples/` for a list of example configuration files. 

TODO: Expand this documentation...need docs for each reconstruction type and parameter.
