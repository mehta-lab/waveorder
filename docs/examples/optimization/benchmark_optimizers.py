"""Benchmark optimization methods on simulated phase 2D data.

Sweeps learning rate for each method to find the best configuration.

Ground truth: z_focus_offset=2, tilt_azimuth=pi/4, tilt_zenith=0.40
Initial guess: all zeros
"""

import os

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch

from waveorder.api import phase
from waveorder.focus import compute_midband_power
from waveorder.models import isotropic_thin_3d
from waveorder.optim import optimize_reconstruction
from waveorder.optim.logging import PrintLogger

# --- Ground truth and simulation ---
gt_z = 2.0
gt_az = np.pi / 4
gt_zen = 0.40

gt_settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        z_focus_offset=gt_z,
        tilt_angle_azimuth=gt_az,
        tilt_angle_zenith=gt_zen,
    )
)
_, data = phase.simulate(
    gt_settings,
    recon_dim=2,
    zyx_shape=(11, 128, 128),
)

# --- Build reconstruct_fn and loss_fn ---
s = gt_settings.transfer_function.resolve_floats()
zyx_data = torch.tensor(data.values[0], dtype=torch.float32)
Z = zyx_data.shape[0]
OPTIM_STEEPNESS = 100.0


def reconstruct_fn(zyx, **params):
    z_offset = params.get("z_focus_offset", s.z_focus_offset)
    tilt_zenith = params.get(
        "tilt_angle_zenith",
        s.tilt_angle_zenith,
    )
    tilt_azimuth = params.get(
        "tilt_angle_azimuth",
        s.tilt_angle_azimuth,
    )

    z_positions = (-torch.arange(Z) + (Z // 2) + z_offset) * s.z_pixel_size
    return isotropic_thin_3d.reconstruct(
        zyx,
        yx_pixel_size=s.yx_pixel_size,
        z_position_list=z_positions,
        wavelength_illumination=s.wavelength_illumination,
        index_of_refraction_media=s.index_of_refraction_media,
        numerical_aperture_illumination=s.numerical_aperture_illumination,
        numerical_aperture_detection=s.numerical_aperture_detection,
        invert_phase_contrast=s.invert_phase_contrast,
        regularization_strength=(gt_settings.apply_inverse.regularization_strength),
        tilt_angle_zenith=tilt_zenith,
        tilt_angle_azimuth=tilt_azimuth,
        pupil_steepness=OPTIM_STEEPNESS,
    )[1]


MIDBAND_FRACTIONS = (0.1, 0.5)


def loss_fn(recon):
    loss = -compute_midband_power(
        recon,
        NA_det=s.numerical_aperture_detection,
        lambda_ill=s.wavelength_illumination,
        pixel_size=s.yx_pixel_size,
        midband_fractions=MIDBAND_FRACTIONS,
    )
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


gt_values = {
    "z_focus_offset": gt_z,
    "tilt_angle_azimuth": gt_az,
    "tilt_angle_zenith": gt_zen,
}

# --- Sweep learning rates for each method ---
learning_rates = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# For grid_search, lr = grid step size
grid_steps = [0.1, 0.25, 0.5, 1.0]

print(f"Data shape: {zyx_data.shape}")
print(f"Ground truth: z={gt_z}, az={gt_az:.4f}, zen={gt_zen}")
print()

# Header
print(
    f"{'method':<14} {'lr':>5} {'time':>6} {'best_loss':>10} {'z_err':>6} {'az_err':>7} {'zen_found':>10} {'iters':>6}"
)
print("-" * 75)

for method in ["adam", "lbfgs", "nelder_mead", "grid_search"]:
    lrs = grid_steps if method == "grid_search" else learning_rates
    max_iter = 300 if method == "nelder_mead" else 50

    for lr in lrs:
        params = {
            "z_focus_offset": (0.0, lr),
            "tilt_angle_azimuth": (0.0, lr),
            "tilt_angle_zenith": (0.0, lr),
        }

        kwargs = dict(
            data=zyx_data,
            reconstruct_fn=reconstruct_fn,
            loss_fn=loss_fn,
            optimizable_params=params,
            method=method,
            logger=PrintLogger(),
        )
        if method != "grid_search":
            kwargs["max_iterations"] = max_iter

        result = optimize_reconstruction(**kwargs)

        total_time = sum(result.wall_times)
        best_loss = min(result.loss_history)
        v = result.optimized_values
        z_err = abs(v["z_focus_offset"] - gt_z)
        az_err = abs(v["tilt_angle_azimuth"] - gt_az)
        zen_val = v["tilt_angle_zenith"]

        print(
            f"{method:<14} {lr:>5.2f} {total_time:>5.1f}s "
            f"{best_loss:>10.3f} {z_err:>6.3f} {az_err:>7.4f} "
            f"{zen_val:>10.4f} {result.iterations_used:>6}"
        )

    print()
