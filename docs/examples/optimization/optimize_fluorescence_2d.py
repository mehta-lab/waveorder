"""Example: optimize fluorescence reconstruction parameters.

Simulates fluorescence data with a known z_focus_offset,
then optimizes from a wrong initial guess to recover it.
"""

import datetime

from waveorder.api import fluorescence
from waveorder.optim import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings

# Ground truth parameters
gt_z_offset = 0.6

# Simulate with ground truth
gt_settings = fluorescence.Settings(
    transfer_function=fluorescence.TransferFunctionSettings(
        z_focus_offset=gt_z_offset,
    )
)
phantom, data = fluorescence.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 256, 256))

# Optimize from wrong initial guess
opt_settings = fluorescence.Settings(
    transfer_function=fluorescence.TransferFunctionSettings(
        z_focus_offset=OptimizableFloat(init=0, lr=0.1),
    )
)

log_dir = f"./runs/{datetime.datetime.now():%Y%m%d_%H%M%S}"

optimized_settings, recon = fluorescence.optimize(
    data,
    settings=opt_settings,
    max_iterations=50,
    loss_settings=MidbandPowerLossSettings(midband_fractions=[0.01, 0.5]),
    log_dir=log_dir,
    log_images=True,
)

s = optimized_settings.transfer_function
print(f"\n{'Parameter':<20} {'Ground truth':>12} {'Optimized':>12}")
print(f"{'z_focus_offset':<20} {gt_z_offset:>12.3f} {s.z_focus_offset:>12.3f}")

print("\nTo view optimization logs, run:")
print("  tensorboard --logdir ./runs")
