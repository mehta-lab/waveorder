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
(`recOrder_session.zip` from zenodo). It is already an **OME-Zarr**, so use
its path directly and skip conversion. Save the script to the working folder,
then:

```bash
uv run get_example_data.py            # ~24 MB download
```

The stack is a single `BF` brightfield channel with 11 defocus slices
(yx = 0.325 µm, z = 2.0 µm), imaged in air. Reconstruct **2D phase** from it
with the draft config in `references/example_qpi_2d.md` (both
`invert_phase_contrast` copies, as Stage 3 describes):

```bash
wo rec -i <raw_data.zarr>/0/0/0 -c invert_false.yml -o ./invert_false.zarr
wo rec -i <raw_data.zarr>/0/0/0 -c invert_true.yml -o ./invert_true.zarr
wo view <raw_data.zarr> ./invert_false.zarr ./invert_true.zarr </dev/null &
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
uv run to_omezarr.py <TIFF_PATH> ./data.zarr \
    --channel-name Brightfield --axes ZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

- `--axes` is the axis order of the *input* array, any subset/order of `TCZYX`
  (e.g. `ZYX`, `CZYX`, `TZYX`). The script transposes/expands to TCZYX.
- `--channel-name` takes one name per channel (repeat the flag). Must match
  `input_channel_names` in the config later.
- Pixel sizes are in micrometers; they set the scale transform used by napari
  and by the reconstruction's optical model.

The script does not read OME-XML metadata, so the axes and pixel sizes must be
given explicitly (the pixel-size flags are required). For a **folder of
tiles**, stitch or pick one tile first; `wo tile-stitch` (`wo ts`) handles
multi-position mosaics once each tile is an OME-Zarr position.

## Case 4: in-memory / `.npy` array or plain (non-OME) zarr

For a plain zarr array, point the converter at the array directory and pass
`--axes` describing its dimension order. For an in-memory array, save to
`.npy` first (`np.save("arr.npy", arr)`), then:

```bash
uv run to_omezarr.py arr.npy ./data.zarr \
    --channel-name GFP --axes CZYX \
    --yx-pixel-size 0.1 --z-pixel-size 0.25
```

## Cropping an ROI

Stages 3 and 4 always run on `roi.zarr`: one position and one time point of
the converted store, cropped further to one biological unit of interest when
that is much smaller than the FOV (tutorial Stage 2b). For a spatial crop,
have the user find the unit in napari (`wo view ./data.zarr </dev/null &`),
hover the cursor over it, and read the coordinates from the status bar. Then:

```bash
uv run crop_roi.py -i ./data.zarr --position 0/0/0 -o ./roi.zarr \
    --t 0:1 --y 800:1312 --x 900:1412        # optionally --z 5:25
```

`-i` takes the plate root (with `--position`, default the first position) or
a position path such as `./data.zarr/0/0/0`. Ranges are `start:stop` in
voxels; omitted axes are kept whole, so always pass `--t` to keep one time
point. Channel names and pixel scale carry over. Verify with
`wo view ./roi.zarr </dev/null &` and iterate on the bounds until the user
confirms the unit is captured.

## Sanity checks

- **Axis order** is the most common mistake. If the reconstruction looks
  scrambled, re-check `--axes`.
- **Z scrolls through focus:** in `wo view`, scrolling the Z slider should
  move the plane of focus through the sample.
- **Pixel sizes are in µm** and roughly match the objective (e.g. a 63x/1.4
  objective on a typical camera gives ~0.1 µm yx).
