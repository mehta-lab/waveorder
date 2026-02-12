"""3D phase reconstruction: API vs model-level comparison."""

import numpy as np
import torch

from waveorder.api import phase
from waveorder.models import phase_thick_3d

# Parameters matching the model-level example (phase_thick_3d.py)
settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        wavelength_illumination=0.532,
        yx_pixel_size=0.1,
        z_pixel_size=0.25,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
        index_of_refraction_media=1.3,
    ),
    apply_inverse=phase.ApplyInverseSettings(
        regularization_strength=1e-3,
    ),
)

# API level: simulate + reconstruct
phantom, data = phase.simulate(
    settings,
    recon_dim=3,
    zyx_shape=(100, 256, 256),
    index_of_refraction_sample=1.50,
)
tf = phase.compute_transfer_function(data, recon_dim=3, settings=settings)
result_api = phase.apply_inverse_transfer_function(
    data, tf, recon_dim=3, settings=settings
)

# Model level: same data, same TF
zyx_data = torch.tensor(data.sel(c="Brightfield").values, dtype=torch.float32)
real_tf = torch.tensor(tf["real_potential_transfer_function"].values)
imag_tf = torch.tensor(tf["imaginary_potential_transfer_function"].values)
zyx_recon_model = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data,
    real_tf,
    imag_tf,
    z_padding=0,
    regularization_strength=1e-3,
)

# Verify model and API match
assert np.allclose(
    result_api.sel(c="Phase3D").values, zyx_recon_model.numpy(), atol=1e-6
)
print("Model-level and API-level outputs match.")

# One-liner
result = phase.reconstruct(data, recon_dim=3, settings=settings)
assert np.allclose(result.values, result_api.values)
print("One-liner matches detailed API.")
