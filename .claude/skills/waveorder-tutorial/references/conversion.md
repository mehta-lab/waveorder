# Converting data to OME-Zarr

`waveorder` reads **OME-Zarr** (HCS layout) with a **TCZYX** array, named
channels, and a pixel-size scale transform. If the data is already OME-Zarr,
no conversion is needed. `scripts/to_omezarr.py` handles the other raw inputs.
Below are the cases and the flags to use.

## Example data (no data of your own)

`scripts/get_example_data.py` downloads the **QPI-from-defocus** sample
(`recOrder_session.zip` from zenodo, the dataset behind
`docs/examples/demos/QPI_defocus`). It is already an **OME-Zarr**, so use its
path directly and skip conversion:

```bash
python scripts/get_example_data.py            # ~10 MB
```

The stack is a single `BF` brightfield channel with 11 defocus slices
(yx = 0.325 µm, z = 2.0 µm), imaged in air. Reconstruct **2D phase** from it
with `references/example_qpi_2d.yml`:

```bash
wo rec -i <raw_data.zarr>/0/0/0 -c references/example_qpi_2d.yml -o ./qpi_2d_recon.zarr
wo view <raw_data.zarr> ./qpi_2d_recon.zarr
```

## The converter

```bash
python scripts/to_omezarr.py <INPUT> <OUTPUT.zarr> \
    --channel-name <NAME> \
    --axes <AXES> \
    --yx-pixel-size <UM> \
    --z-pixel-size <UM>
```

- `<INPUT>` — a `.tif`/`.tiff`/`.ome.tif`, a `.npy`, or a plain zarr array dir.
- `--axes` — the axis order of the *input* array, any subset/order of `TCZYX`
  (e.g. `ZYX`, `CZYX`, `YX`, `TZYX`). The script transposes/expands to TCZYX.
- `--channel-name` — one name per channel (repeat the flag for multiple).
  Must match `input_channel_names` in the config later.
- Pixel sizes in micrometers; used for the scale transform (and thus napari's
  physical axes and the reconstruction's optical model).

Verify with `wo view <OUTPUT.zarr>` afterwards.

## Case 1 — OME-Zarr (preferred)

Already in `waveorder`'s native format — **don't convert**. Ask the user to
paste the path. Confirm axis order and channel names with `wo view <path>`; if
it opens and scrolls in Z correctly, use it directly.

## Case 2 — TIFF / OME-TIFF

Ask the user to paste the path to their TIFF (a single file or a folder of
tiles). Single 3D stack saved ZYX:
```bash
python scripts/to_omezarr.py <TIFF_PATH> ./data.zarr \
    --channel-name Brightfield --axes ZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```
An OME-TIFF usually already encodes axes and scale — the script reads them when
present, and the flags override. For a **folder of tiles**, stitch or pick one
tile first; `wo tile-stitch` (`wo ts`) handles multi-position mosaics once each
tile is an OME-Zarr position.

## Case 3 — in-memory / `.npy` array or plain (non-OME) zarr

For a plain zarr array, point the converter at the array directory and pass
`--axes` describing its dimension order. For an in-memory array, save to `.npy`
first.

Save to `.npy` first (`np.save("arr.npy", arr)`), then:
```bash
python scripts/to_omezarr.py arr.npy ./data.zarr \
    --channel-name GFP --axes CZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

## Sanity checks

- **Axis order** is the most common mistake — if the reconstruction looks
  scrambled, re-check `--axes`.
- **Z scrolls through focus:** in `wo view`, scrolling the Z slider should move
  the plane of focus through the sample.
- **Pixel sizes are in µm** and roughly match the objective (e.g. a 63×/1.4
  objective on a typical camera → ~0.1 µm yx).
