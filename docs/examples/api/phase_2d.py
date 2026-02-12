"""2D phase reconstruction: API vs model-level comparison."""

import numpy as np
import torch

from waveorder.api import phase
from waveorder.models import isotropic_thin_3d

# Parameters matching the model-level example (isotropic_thin_3d.py)
settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        wavelength_illumination=0.532,
        yx_pixel_size=0.1,
        z_pixel_size=0.25,
        z_focus_offset=0,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
        index_of_refraction_media=1.3,
    ),
    apply_inverse=phase.ApplyInverseSettings(
        regularization_strength=1e-2,
    ),
)

# API level: simulate + reconstruct
phantom, data = phase.simulate(
    settings,
    recon_dim=2,
    zyx_shape=(100, 256, 256),
    index_of_refraction_sample=1.33,
)
tf = phase.compute_transfer_function(data, recon_dim=2, settings=settings)
result_api = phase.apply_inverse_transfer_function(
    data, tf, recon_dim=2, settings=settings
)

# Model level: same data, same TF
zyx_data = torch.tensor(data.sel(c="Brightfield").values, dtype=torch.float32)
singular_system = (
    torch.tensor(tf["singular_system_U"].values),
    torch.tensor(tf["singular_system_S"].values),
    torch.tensor(tf["singular_system_Vh"].values),
)
_, yx_phase_recon = isotropic_thin_3d.apply_inverse_transfer_function(
    zyx_data,
    singular_system,
    regularization_strength=1e-2,
)

# Verify model and API match
assert np.allclose(
    result_api.sel(c="Phase2D").values[0], yx_phase_recon.numpy(), atol=1e-6
)
print("Model-level and API-level outputs match.")

# One-liner
result = phase.reconstruct(data, recon_dim=2, settings=settings)
assert np.allclose(result.values, result_api.values)
print("One-liner matches detailed API.")
