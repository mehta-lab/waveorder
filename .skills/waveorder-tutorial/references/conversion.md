# Converting data to OME-Zarr

`waveorder` reads **OME-Zarr** (HCS layout) with a **TCZYX** array, named
channels, and a pixel-size scale transform. If the data is already OME-Zarr,
no conversion is needed. Route by input format:

| input format                         | tool                        |
|--------------------------------------|-----------------------------|
| OME-Zarr                             | none (verify with `wo view`) |
| Micro-Manager TIFF dataset           | `iohub convert`             |
| bare TIFF / OME-TIFF / npy / plain zarr | `scripts/to_omezarr.py`  |
| large OME-Zarr needing a small ROI   | `scripts/crop_roi.py`       |

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

## Case 1: OME-Zarr (preferred)

Already in `waveorder`'s native format, so **don't convert**. Ask the user to
paste the path. Confirm axis order and channel names with `wo view <path>`; if
it opens and scrolls in Z correctly, use it directly.

## Case 2: Micro-Manager TIFF dataset

The `iohub` package (installed with waveorder) converts Micro-Manager TIFF
datasets (including NDTiff and OME-TIFF MM sessions) reliably, preserving
positions, channel names, and scale metadata:

```bash
iohub convert -i <micromanager_dataset_dir> -o ./data.zarr
```

Prefer this over the generic script whenever the data came out of
Micro-Manager.

## Case 3: bare TIFF / OME-TIFF

Ask the user to **paste the path to their TIFF** (a single file, or a folder
of tiles). Then convert with `scripts/to_omezarr.py`:

```bash
python scripts/to_omezarr.py <TIFF_PATH> ./data.zarr \
    --channel-name Brightfield --axes ZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

- `--axes` is the axis order of the *input* array, any subset/order of `TCZYX`
  (e.g. `ZYX`, `CZYX`, `TZYX`). The script transposes/expands to TCZYX.
- `--channel-name` takes one name per channel (repeat the flag). Must match
  `input_channel_names` in the config later.
- Pixel sizes are in micrometers; they set the scale transform used by napari
  and by the reconstruction's optical model.

An OME-TIFF usually already encodes axes and scale; the script reads them when
present, and the flags override. For a **folder of tiles**, stitch or pick one
tile first; `wo tile-stitch` (`wo ts`) handles multi-position mosaics once
each tile is an OME-Zarr position.

## Case 4: in-memory / `.npy` array or plain (non-OME) zarr

For a plain zarr array, point the converter at the array directory and pass
`--axes` describing its dimension order. For an in-memory array, save to
`.npy` first (`np.save("arr.npy", arr)`), then:

```bash
python scripts/to_omezarr.py arr.npy ./data.zarr \
    --channel-name GFP --axes CZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

## Cropping an ROI

To iterate quickly, crop the converted store down to one biological unit of
interest (tutorial Stage 2b). Have the user find the unit in napari
(`wo view ./data.zarr &`), hover the cursor over it, and read the coordinates
from the status bar. Then:

```bash
python scripts/crop_roi.py -i ./data.zarr -o ./roi.zarr \
    --y 800:1312 --x 900:1412        # optionally --z 5:25, --t 0:1
```

Ranges are `start:stop` in voxels; omitted axes are kept whole. Channel names
and pixel scale carry over. Verify with `wo view ./roi.zarr &` and iterate on
the bounds until the user confirms the unit is captured.

## Sanity checks

- **Axis order** is the most common mistake. If the reconstruction looks
  scrambled, re-check `--axes`.
- **Z scrolls through focus:** in `wo view`, scrolling the Z slider should
  move the plane of focus through the sample.
- **Pixel sizes are in µm** and roughly match the objective (e.g. a 63x/1.4
  objective on a typical camera gives ~0.1 µm yx).
