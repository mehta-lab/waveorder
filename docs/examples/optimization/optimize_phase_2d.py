"""Example: optimize phase reconstruction parameters.

Simulates brightfield phase data with known z_focus_offset and
tilt illumination, then optimizes from (0, 0, 0) to recover
the ground truth parameters.
"""

import datetime

import numpy as np

from waveorder.api import phase
from waveorder.optim import OptimizableFloat

# To use your own data instead of simulated data, create a CZYX xr.DataArray:
#
#   import xarray as xr
#
#   zyx_array = np.load("my_data.npy")      # shape (Z, Y, X)
#   data = xr.DataArray(
#       zyx_array[None],                    # add C dimension -> (1, Z, Y, X)
#       dims=("c", "z", "y", "x"),
#   )
#
# Then pass `data` to `phase.optimize(data, ...)` below.

# Ground truth parameters
gt_z_offset = 0.6
gt_tilt_zenith = 0.5
gt_tilt_azimuth = np.pi / 4

# Simulate with ground truth
gt_settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        z_focus_offset=gt_z_offset,
        tilt_angle_zenith=gt_tilt_zenith,
        tilt_angle_azimuth=gt_tilt_azimuth,
    )
)
phantom, data = phase.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 256, 256))

# Optimize from (0, 0, 0) initial guess
opt_settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        z_focus_offset=OptimizableFloat(init=0, lr=0.1),
        tilt_angle_zenith=OptimizableFloat(init=0, lr=0.1),
        tilt_angle_azimuth=OptimizableFloat(init=0, lr=0.1),
    )
)

log_dir = f"./runs/{datetime.datetime.now():%Y%m%d_%H%M%S}"

optimized_settings, recon = phase.optimize(
    data,
    settings=opt_settings,
    max_iterations=50,
    midband_fractions=(0.1, 0.5),
    log_dir=log_dir,
    log_images=True,
)

s = optimized_settings.transfer_function
print(f"\n{'Parameter':<20} {'Ground truth':>12} {'Optimized':>12}")
print(f"{'z_focus_offset':<20} {gt_z_offset:>12.3f} {s.z_focus_offset:>12.3f}")
print(f"{'tilt_angle_zenith':<20} {gt_tilt_zenith:>12.3f} {s.tilt_angle_zenith:>12.3f}")
print(f"{'tilt_angle_azimuth':<20} {gt_tilt_azimuth:>12.3f} {s.tilt_angle_azimuth:>12.3f}")

print("\nTo view optimization logs and images, run TensorBoard in your terminal:")
print("  tensorboard --logdir ./runs")
print("Then open the displayed URL in your browser.")
