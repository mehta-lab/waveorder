"""Atomic benchmark stages for GPU reconstruction.

Each function isolates one unit of work so it can be independently
benchmarked with torch.utils.benchmark.Timer.
"""

import torch
import xarray as xr

from config import PHYSICS_20X
from waveorder.api import phase
from waveorder.api._utils import _to_singular_system, _wrap_output_tensor
from waveorder.models import isotropic_thin_3d
from waveorder.optim.losses import MidbandPowerLossSettings, build_loss_fn

# ---------------------------------------------------------------------------
# 1. TF compute (CPU/GPU — includes SVD)
# ---------------------------------------------------------------------------

def compute_tf(
    reference_tile: xr.DataArray,
    settings: phase.Settings,
    device: str,
) -> xr.Dataset:
    """Compute transfer function for a tile shape."""
    return phase.compute_transfer_function(
        reference_tile, recon_dim=2, settings=settings, device=device,
    )


# ---------------------------------------------------------------------------
# 2. TF xr.Dataset → CPU tensors (numpy → torch)
# ---------------------------------------------------------------------------

def tf_to_tensors(tf: xr.Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (U, S, Vh) singular system from TF dataset as CPU tensors."""
    return _to_singular_system(tf)


# ---------------------------------------------------------------------------
# 3. TF tensors → GPU (H2D)
# ---------------------------------------------------------------------------

def tf_to_device(
    singular_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer (U, S, Vh) to GPU."""
    U, S, Vh = singular_system
    return U.to(device), S.to(device), Vh.to(device)


# ---------------------------------------------------------------------------
# 4. Tiles: xr.DataArray list → CPU tensors (numpy → torch)
# ---------------------------------------------------------------------------

def tiles_to_tensors(tiles: list[xr.DataArray]) -> list[torch.Tensor]:
    """Convert list of CZYX xr.DataArrays to list of ZYX CPU tensors."""
    return [torch.tensor(d.values[0], dtype=torch.float32) for d in tiles]


# ---------------------------------------------------------------------------
# 5. Tiles: CPU tensors → GPU + stack into batch (H2D + stack)
# ---------------------------------------------------------------------------

def tiles_to_device(
    tensors: list[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Transfer tile tensors to GPU and stack into (B, Z, Y, X)."""
    if len(tensors) == 1:
        return tensors[0].to(device)  # (Z, Y, X)
    return torch.stack([t.to(device) for t in tensors])  # (B, Z, Y, X)


# ---------------------------------------------------------------------------
# 6. GPU reconstruct (pure compute, everything already on device)
# ---------------------------------------------------------------------------

def reconstruct_on_device(
    zyx_tensor: torch.Tensor,
    singular_system_device: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    settings: phase.Settings,
) -> torch.Tensor:
    """Run isotropic_thin_3d.apply_inverse_transfer_function on GPU."""
    _, output = isotropic_thin_3d.apply_inverse_transfer_function(
        zyx_tensor,
        singular_system_device,
        **settings.apply_inverse.model_dump(),
    )
    return output


# ---------------------------------------------------------------------------
# 7. D2H: GPU tensor → CPU
# ---------------------------------------------------------------------------

def result_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Transfer result tensor from GPU to CPU."""
    return tensor.detach().cpu()


# ---------------------------------------------------------------------------
# 8. Wrap output: CPU tensor → xr.DataArray(s)
# ---------------------------------------------------------------------------

def wrap_output(
    output: torch.Tensor,
    tiles: list[xr.DataArray],
    recon_dim: int = 2,
) -> list[xr.DataArray]:
    """Wrap GPU output back into xr.DataArrays."""
    from waveorder.api.phase import _output_channel_names
    ch = _output_channel_names(recon_phase=True, recon_dim=recon_dim)
    sz = recon_dim == 2
    if output.ndim == 2:
        # Single tile: (Y, X)
        return [_wrap_output_tensor(output, ch, tiles[0], sz)]
    # Batched: (B, Y, X)
    return [_wrap_output_tensor(output[i], ch, tiles[i], sz) for i in range(len(tiles))]


# ===========================================================================
# Forward / backward per-iteration profiling
# ===========================================================================

def _make_z_positions(Z: int, z_focus_offset, device):
    """Build z_position_list on device, matching phase.optimize's convention."""
    p = PHYSICS_20X
    z_indices = -torch.arange(Z, device=device) + (Z // 2)
    return (z_indices + z_focus_offset) * p["z_pixel_size"]


def forward_pass(data: torch.Tensor, device: str, svd_backend: str = "closed_form") -> torch.Tensor:
    """Full forward pass: TF compute + SVD + inverse filter.

    Calls isotropic_thin_3d.reconstruct() with grad-enabled tilt params.
    data: (Z,Y,X) or (B,Z,Y,X) on device.
    Returns phase reconstruction tensor (still in autograd graph).
    """
    p = PHYSICS_20X
    Z = data.shape[-3]
    B = data.shape[0] if data.ndim == 4 else 1

    z_offset = torch.tensor(0.1, device=device, requires_grad=True)
    tilt_zen = torch.tensor([0.1] * B, device=device, requires_grad=True) if B > 1 else torch.tensor(0.1, device=device, requires_grad=True)
    tilt_azi = torch.tensor([0.1] * B, device=device, requires_grad=True) if B > 1 else torch.tensor(0.1, device=device, requires_grad=True)

    z_positions = _make_z_positions(Z, z_offset, device)

    _, phase_recon = isotropic_thin_3d.reconstruct(
        data,
        yx_pixel_size=p["yx_pixel_size"],
        z_position_list=z_positions,
        wavelength_illumination=p["wavelength_illumination"],
        index_of_refraction_media=p["index_of_refraction_media"],
        numerical_aperture_illumination=p["numerical_aperture_illumination"],
        numerical_aperture_detection=p["numerical_aperture_detection"],
        invert_phase_contrast=p["invert_phase_contrast"],
        regularization_strength=p["regularization_strength"],
        tilt_angle_zenith=tilt_zen,
        tilt_angle_azimuth=tilt_azi,
        pupil_steepness=p["pupil_steepness"],
        svd_backend=svd_backend,
    )
    return phase_recon


def backward_pass(loss: torch.Tensor):
    """Backward pass: compute gradients through the forward graph."""
    loss.backward(retain_graph=True)


def forward_and_loss(data: torch.Tensor, loss_fn, device: str, svd_backend: str = "closed_form"):
    """Forward + loss computation (returns loss tensor for backward)."""
    recon = forward_pass(data, device, svd_backend=svd_backend)
    if recon.ndim == 2:
        return loss_fn(recon)
    # Batched: sum per-tile losses
    return sum(loss_fn(recon[b]) for b in range(recon.shape[0]))


def full_iteration(data, optimizer, loss_fn, reconstruct_fn):
    """One complete optimization iteration: zero_grad → forward → loss → backward → step."""
    optimizer.zero_grad()
    recon = reconstruct_fn(data)
    if recon.ndim == 2:
        loss = loss_fn(recon)
    else:
        loss = sum(loss_fn(recon[b]) for b in range(recon.shape[0]))
    loss.backward()
    optimizer.step()
    return loss


# --- Forward substages (called independently for profiling) ---

def bench_calculate_tf(yx_shape, z_positions, tilt_zenith, tilt_azimuth):
    """Just the TF computation (pupil + WOTF)."""
    p = PHYSICS_20X
    return isotropic_thin_3d.calculate_transfer_function(
        yx_shape=yx_shape,
        yx_pixel_size=p["yx_pixel_size"],
        z_position_list=z_positions,
        wavelength_illumination=p["wavelength_illumination"],
        index_of_refraction_media=p["index_of_refraction_media"],
        numerical_aperture_illumination=p["numerical_aperture_illumination"],
        numerical_aperture_detection=p["numerical_aperture_detection"],
        invert_phase_contrast=p["invert_phase_contrast"],
        tilt_angle_zenith=tilt_zenith,
        tilt_angle_azimuth=tilt_azimuth,
        pupil_steepness=p["pupil_steepness"],
    )


def bench_calculate_svd(absorption_tf, phase_tf, svd_backend="closed_form"):
    """Just the SVD computation."""
    return isotropic_thin_3d.calculate_singular_system(absorption_tf, phase_tf, svd_backend=svd_backend)


def bench_apply_inverse(zyx_data, singular_system):
    """Just the Tikhonov inverse filter application."""
    p = PHYSICS_20X
    return isotropic_thin_3d.apply_inverse_transfer_function(
        zyx_data,
        singular_system,
        regularization_strength=p["regularization_strength"],
    )


def make_loss_fn():
    """Build the same loss function used in optimization."""
    p = PHYSICS_20X
    return build_loss_fn(
        MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
        NA_det=p["numerical_aperture_detection"],
        wavelength=p["wavelength_illumination"],
        pixel_size=p["yx_pixel_size"],
    )


# ---------------------------------------------------------------------------
# 9. Full optimization loop (TF compute + reconstruct + loss, N iterations)
# ---------------------------------------------------------------------------

def optimize_tile(
    tile: xr.DataArray,
    max_iterations: int,
    device: str,
):
    """Run phase.optimize on a single tile (the real per-tile workload)."""
    from waveorder.optim import OptimizableFloat
    from waveorder.optim.losses import MidbandPowerLossSettings

    p = PHYSICS_20X
    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=p["wavelength_illumination"],
            yx_pixel_size=p["yx_pixel_size"],
            z_pixel_size=p["z_pixel_size"],
            z_focus_offset=OptimizableFloat(init=0.1, lr=0.02),
            numerical_aperture_illumination=p["numerical_aperture_illumination"],
            numerical_aperture_detection=p["numerical_aperture_detection"],
            index_of_refraction_media=p["index_of_refraction_media"],
            invert_phase_contrast=p["invert_phase_contrast"],
            tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.02),
            tilt_angle_azimuth=OptimizableFloat(init=0.1, lr=0.02),
        ),
        apply_inverse=phase.ApplyInverseSettings(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=p["regularization_strength"],
        ),
    )

    return phase.optimize(
        tile,
        recon_dim=2,
        settings=opt_settings,
        max_iterations=max_iterations,
        method="adam",
        convergence_tol=None,
        convergence_patience=None,
        loss_settings=MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
        device=device,
    )


# ---------------------------------------------------------------------------
# 10. Batched optimization (B tiles with per-tile params, on GPU)
# ---------------------------------------------------------------------------

def optimize_batch(
    tiles: list[xr.DataArray],
    max_iterations: int,
    device: str,
):
    """Run batched optimization: stack tiles into (B,Z,Y,X), optimize with per-tile params.

    Calls optimize_reconstruction directly with a (B,Z,Y,X) tensor so each
    tile gets independent (B,) shaped parameter tensors optimized in parallel.
    """
    import torch

    from waveorder.device import resolve_device
    from waveorder.models import isotropic_thin_3d
    from waveorder.optim.losses import MidbandPowerLossSettings, build_loss_fn
    from waveorder.optim.optimize import optimize_reconstruction

    device = resolve_device(device)

    # Stack tiles into (B, Z, Y, X)
    zyx_batch = torch.stack(
        [torch.tensor(t.values[0], dtype=torch.float32, device=device) for t in tiles]
    )

    B = zyx_batch.shape[0]
    Z = zyx_batch.shape[1]

    # Physics params (20x OPS pipeline)
    p = PHYSICS_20X
    yx_pixel_size = p["yx_pixel_size"]
    z_pixel_size = p["z_pixel_size"]
    wavelength = p["wavelength_illumination"]
    n_media = p["index_of_refraction_media"]
    na_ill = p["numerical_aperture_illumination"]
    na_det = p["numerical_aperture_detection"]
    reg_strength = p["regularization_strength"]
    optim_steepness = p["pupil_steepness"]

    z_indices = -torch.arange(Z, device=device) + (Z // 2)

    def reconstruct_fn(data, **tensor_params):
        z_offset = tensor_params.get("z_focus_offset", 0.1)
        tilt_zenith = tensor_params.get("tilt_angle_zenith", 0.1)
        tilt_azimuth = tensor_params.get("tilt_angle_azimuth", 0.1)

        # Ensure tensors are on the correct device
        if isinstance(z_offset, torch.Tensor) and z_offset.device.type == "cpu":
            z_offset = z_offset.to(device)
        if isinstance(tilt_zenith, torch.Tensor) and tilt_zenith.device.type == "cpu":
            tilt_zenith = tilt_zenith.to(device)
        if isinstance(tilt_azimuth, torch.Tensor) and tilt_azimuth.device.type == "cpu":
            tilt_azimuth = tilt_azimuth.to(device)

        # z_positions must be (Z,) — shared across batch.
        # When batched, z_offset is (B,) but reconstruct only supports shared z_positions.
        # Use mean of batch z_offsets as shared value.
        if isinstance(z_offset, torch.Tensor) and z_offset.ndim >= 1:
            z_offset = z_offset.mean()
        z_positions = (z_indices + z_offset) * z_pixel_size

        return isotropic_thin_3d.reconstruct(
            data,
            yx_pixel_size=yx_pixel_size,
            z_position_list=z_positions,
            wavelength_illumination=wavelength,
            index_of_refraction_media=n_media,
            numerical_aperture_illumination=na_ill,
            numerical_aperture_detection=na_det,
            invert_phase_contrast=False,
            regularization_strength=reg_strength,
            tilt_angle_zenith=tilt_zenith,
            tilt_angle_azimuth=tilt_azimuth,
            pupil_steepness=optim_steepness,
        )[1]  # phase only

    loss_fn = build_loss_fn(
        MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
        NA_det=na_det,
        wavelength=wavelength,
        pixel_size=yx_pixel_size,
    )

    opt_params = {
        "z_focus_offset": (0.1, 0.01),
        "tilt_angle_zenith": (0.1, 0.005),
        "tilt_angle_azimuth": (0.1, 0.001),
    }

    return optimize_reconstruction(
        data=zyx_batch,
        reconstruct_fn=reconstruct_fn,
        loss_fn=loss_fn,
        optimizable_params=opt_params,
        max_iterations=max_iterations,
        method="adam",
        use_gradients=True,
        convergence_tol=None,
        convergence_patience=None,
    )


# ---------------------------------------------------------------------------
# Composite helpers (for end-to-end benchmarks)
# ---------------------------------------------------------------------------

def tf_and_reconstruct(
    tiles: list[xr.DataArray],
    reference_tile: xr.DataArray,
    settings: phase.Settings,
    device: str,
):
    """Full GPU workload: TF compute + reconstruct (for warmup / e2e)."""
    tf = compute_tf(reference_tile, settings, device)
    if len(tiles) == 1:
        return phase.apply_inverse_transfer_function(
            tiles[0], tf, recon_dim=2, settings=settings, device=device,
        )
    return phase.apply_inverse_transfer_function(
        tiles, tf, recon_dim=2, settings=settings, device=device,
    )
