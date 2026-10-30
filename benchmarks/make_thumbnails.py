"""Build the raw/recon thumbnail grid in benchmarks/benchmark_datasets.png.

Two rows (raw input, reconstruction), one column per benchmark case,
from a ``wo bm run --scope all --save-all`` run. Each case directory is
self-describing: ``cli_command.sh`` holds the exact input position path
(cropped input when the case crops) and ``config.yml`` names the input
channel.

Usage::

    python benchmarks/make_thumbnails.py [RUN_DIR] [OUT_PNG]

``RUN_DIR`` defaults to the newest ``*_regression`` run under
``$WAVEORDER_BENCH_OUTPUT``; ``OUT_PNG`` defaults to
``benchmark_datasets.png``. The 3D panels show the highest-contrast
z-slice; flat-filled borders (e.g. sheared oblique registrations) are
trimmed with one shared window per column so raw and recon stay
pixel-aligned.
"""

import glob
import os
import re
import shlex
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from iohub import open_ome_zarr
from PIL import Image

plt.rcParams["font.family"] = "monospace"

OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else "benchmark_datasets.png"

SHORT_NAMES = {
    "phase_3d_beads": "beads phase 3d",
    "fluor_3d_beads": "beads fluor 3d",
    "shift_variant_grid_beads": "shift-var grid",
    "ops_pheno_center": "ops pheno 2d",
    "ops_pheno_center_opt": "ops pheno 2d opt",
    "ops_pheno_phase_3d": "ops pheno 3d",
    "ops_track_phase_2d": "ops track 2d",
    "ops_track_phase_3d": "ops track 3d",
    "ops_fluor_gfp": "ops fluor gfp",
    "kazansky_birefringence_3d": "kazansky biref",
    "a549_phase_2d": "a549 phase 2d",
    "a549_phase_3d": "a549 phase 3d",
    "cardiomyocyte_phase_3d": "cardio phase 3d",
    "zebrafish_daxi_phase_3d": "zebrafish phase",
    "ev71_phase_3d": "ev71 phase 3d",
    "ev71_fluor_fitc_3d": "ev71 fluor 3d",
    "a549_fluor_3d": "a549 fluor 3d",
    "neuromast_epi_phase_3d": "neuromast epi",
    "neuromast_ls_phase_3d": "neuromast ls",
    "neuromast_tile_phase_3d": "neuromast ls tiled",
    "mantis_v1_neuromast_phase_3d": "neuromast v1",
}


def middle_slice(czyx, c_idx):
    """Return the highest-contrast z-slice of channel ``c_idx`` from a CZYX array."""
    zyx = np.asarray(czyx[c_idx])
    stds = [zyx[z].std() for z in range(zyx.shape[0])]
    return zyx[int(np.argmax(stds))]


def flat_border_slices(img):
    """Return (y, x) slices that trim flat-filled border rows/columns.

    Registered oblique acquisitions carry sheared average-filled bands
    (e.g. the neuromast LS channels); rows/cols whose std is far below
    the median are trimmed. Computed from the RAW image and applied to
    both rows of a column so raw and recon stay pixel-aligned.
    """
    slices = []
    for axis in (0, 1):
        std = img.std(axis=1 - axis)
        valid = np.where(std > 0.2 * np.median(std))[0]
        if len(valid):
            slices.append(slice(valid[0], valid[-1] + 1))
        else:
            slices.append(slice(None))
    return tuple(slices)


def square_crop(img):
    """Center-crop a 2D image to a square."""
    y, x = img.shape
    s = min(y, x)
    y0, x0 = (y - s) // 2, (x - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


STRETCH_OVERRIDES = {
    "phase_3d_beads": (1, 99.9),
    "fluor_3d_beads": (1, 99.9),
    "shift_variant_grid_beads": (1, 99.9),
}


def thumbnail(img, size=256, stretch=(5, 95)):
    """Block-average a square image down to ~size, then percentile-stretch.

    Downsampling before the stretch means the percentiles reflect the
    pixels actually displayed, so fine low-amplitude texture (e.g. 3D
    phase) is not washed out by full-resolution outliers.
    """
    f = max(1, img.shape[0] // size)
    m = (img.shape[0] // f) * f
    small = img[:m, :m].reshape(m // f, f, m // f, f).mean(axis=(1, 3))
    lo, hi = np.percentile(small, list(stretch))
    if hi <= lo:
        return np.zeros_like(small)
    return np.clip((small - lo) / (hi - lo), 0, 1)


def load_position(path):
    """Open an OME-Zarr position path and return (CZYX t=0, channel names)."""
    pos = open_ome_zarr(path, layout="fov", mode="r")
    return np.array(pos["0"][0]), list(pos.channel_names)


def load_first_position(path):
    """Open an OME-Zarr plate or position and return (CZYX t=0, channel names)."""
    ds = open_ome_zarr(path, layout="auto", mode="r")
    try:
        pos = list(ds.positions())[0][1]
    except AttributeError:
        pos = ds
    return np.array(pos["0"][0]), list(pos.channel_names)


if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    root = os.environ.get("WAVEORDER_BENCH_OUTPUT", "/hpc/projects/waveorder/benchmarks")
    run_dir = sorted(glob.glob(f"{root}/*_regression"))[-1]
print("run:", run_dir)

with open(f"{run_dir}/experiment.yml") as f:
    case_order = list(yaml.safe_load(f)["cases"])

panels = []
case_ids = []
for case in case_order:
    case_dir = f"{run_dir}/cases/{case}"
    raw = recon = None
    try:
        try:
            cli = open(f"{case_dir}/cli_command.sh").read()
            input_path = shlex.split(re.search(r"wo (?:rec|tile-stitch) (.*)", cli).group(1))[1]
            with open(f"{case_dir}/config.yml") as f:
                cfg = yaml.safe_load(f)
            chan = cfg.get("recon", cfg)["input_channel_names"][0]
            czyx, names = load_position(input_path)
        except FileNotFoundError:
            # Recovery-type cases skip wo rec; show the simulated input.
            czyx, names = load_first_position(f"{case_dir}/simulated.zarr")
            chan = names[0]
        raw = middle_slice(czyx, names.index(chan))
    except Exception as e:
        print(f"{case}: raw failed: {e}")
    try:
        czyx, names = load_first_position(f"{case_dir}/reconstruction.zarr")
        recon = middle_slice(czyx, 0)
    except Exception as e:
        print(f"{case}: recon failed: {e}")
    panels.append((SHORT_NAMES.get(case, case), raw, recon))
    case_ids.append(case)
    print(f"{case}: raw={None if raw is None else raw.shape} recon={None if recon is None else recon.shape}")

n = len(panels)
fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 3.6))
case_names = case_ids
for j, (name, raw, recon) in enumerate(panels):
    ref = raw if raw is not None else recon
    trim = flat_border_slices(ref) if ref is not None else (slice(None), slice(None))
    for i, img in enumerate([raw, recon]):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        if img is not None:
            stretch = STRETCH_OVERRIDES.get(case_names[j], (5, 95))
            view = img[trim] if img.shape == ref.shape else img
            ax.imshow(thumbnail(square_crop(view), stretch=stretch), cmap="gray", interpolation="auto")
        else:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
    axes[0, j].set_title(name, fontsize=9, rotation=45, ha="left")
axes[0, 0].set_ylabel("raw", fontsize=11)
axes[1, 0].set_ylabel("recon", fontsize=11)
fig.subplots_adjust(left=0.02, right=0.96, top=0.72, bottom=0.02, wspace=0.05, hspace=0.05)
fig.savefig(OUT_PNG, dpi=130)
Image.open(OUT_PNG).convert("L").save(OUT_PNG, optimize=True)
print("wrote", OUT_PNG)
