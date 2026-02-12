"""Birefringence reconstruction: API vs model-level comparison."""

import numpy as np
import torch

from waveorder.api import birefringence
from waveorder.api._utils import radians_to_nanometers
from waveorder.models import inplane_oriented_thick_pol3d

# Parameters matching the model-level example (inplane_oriented_thick_pol3d.py)
settings = birefringence.Settings(
    transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
    apply_inverse=birefringence.ApplyInverseSettings(
        wavelength_illumination=0.532,
    ),
)

# API level: simulate + reconstruct
channel_names = [f"State{i}" for i in range(4)]
phantom, data = birefringence.simulate(
    settings,
    yx_shape=(256, 256),
    scheme="4-State",
)
tf = birefringence.compute_transfer_function(data, settings, channel_names)
result_api = birefringence.apply_inverse_transfer_function(
    data, tf, recon_dim=3, settings=settings
)

# Model level: same data, same TF
intensity_to_stokes = torch.tensor(tf["intensity_to_stokes_matrix"].values)
# Use a single Z slice (birefringence is 2D physics tiled along Z)
czyx_tensor = torch.tensor(data.values[:, :1], dtype=torch.float32)
retardance, orientation, transmittance, depolarization = (
    inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
        czyx_tensor,
        intensity_to_stokes,
    )
)
# API converts retardance from radians to nanometers
retardance_nm = radians_to_nanometers(retardance, 0.532)

# Verify model and API match (compare single Z slice)
assert np.allclose(
    result_api.sel(c="Retardance").values[:1],
    retardance_nm.numpy(),
    atol=1e-5,
)
assert np.allclose(
    result_api.sel(c="Orientation").values[:1],
    orientation.numpy(),
    atol=1e-5,
)
print("Model-level and API-level outputs match.")

# One-liner
result = birefringence.reconstruct(data, settings, channel_names)
assert np.allclose(result.values, result_api.values)
print("One-liner matches detailed API.")
