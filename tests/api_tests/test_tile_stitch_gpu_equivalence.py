"""GPU equivalence — tile-stitch reconstruction on CPU vs CUDA.

Gated by ``@pytest.mark.gpu``: skipped on CPU-only CI runners; run
manually on Bruno gpu node or via the self-hosted GPU runner once it
lands. Tolerance is derived from f32 accumulation error scaled by
the number of contributors at the worst-case interior cell.
"""

import numpy as np
import pytest
import xarray as xr

from waveorder.api.phase import Settings as PhaseSettings
from waveorder.api.tile_stitch import (
    BlendSettings,
    TileSettings,
    TileStitchSettings,
    clear_transfer_function_cache,
    prepare_transfer_function,
)
from waveorder.tile_stitch._engine import tile_stitch_reconstruction

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _isolate_tf_cache():
    clear_transfer_function_cache()
    yield
    clear_transfer_function_cache()


def _have_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _have_cuda(), reason="CUDA device required")
def test_tile_stitch_phase_cpu_vs_cuda_equivalence():
    """Phase tile-stitch on CPU and CUDA produces equivalent output volumes."""
    rng = np.random.default_rng(7)
    data_np = rng.normal(loc=1.0, scale=0.05, size=(1, 8, 32, 32)).astype(np.float32)
    data = xr.DataArray(data_np, dims=("c", "z", "y", "x"))

    settings = TileStitchSettings(
        tile=TileSettings(
            tile_size={"z": 8, "y": 16, "x": 16},
            overlap={"y": 4, "x": 4},
        ),
        blend=BlendSettings(kind="uniform_mean"),
        recon=PhaseSettings(),
    )

    tf_cpu = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    out_cpu = tile_stitch_reconstruction(data, settings, transfer_function=tf_cpu, recon_dim=3, device="cpu")

    clear_transfer_function_cache()
    tf_gpu = prepare_transfer_function(settings, recon_dim=3, device="cuda")
    out_gpu = tile_stitch_reconstruction(data, settings, transfer_function=tf_gpu, recon_dim=3, device="cuda")

    cpu = np.asarray(out_cpu.values)
    gpu = np.asarray(out_gpu.values)

    # atol-from-derivation: phase recon involves an FFT (depth ≈ log2(N))
    # on the tile, Tikhonov inverse, IFFT, and a tile blend reduction.
    # Each f32 floating-point op accumulates ~eps relative error; the
    # cumulative bound for a single recon is ~eps × O(N log N) which
    # for 32³ tiles is roughly eps × 50. Blending adds another factor
    # of ≈ N_contributors. Together, that's ~eps × 200 × peak_magnitude.
    eps32 = np.finfo(np.float32).eps
    n_contrib_worst = 4
    fft_depth_factor = 50
    atol = eps32 * fft_depth_factor * n_contrib_worst * float(np.max(np.abs(cpu)))
    np.testing.assert_allclose(gpu, cpu, atol=atol, rtol=1e-5)
