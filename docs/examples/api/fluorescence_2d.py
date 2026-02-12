"""2D fluorescence deconvolution: API vs model-level comparison."""

import numpy as np
import torch

from waveorder.api import fluorescence
from waveorder.models import isotropic_fluorescent_thin_3d

# Parameters matching the model-level example (isotropic_fluorescent_thin_3d.py)
settings = fluorescence.Settings(
    transfer_function=fluorescence.TransferFunctionSettings(
        yx_pixel_size=0.1,
        z_pixel_size=0.25,
        z_focus_offset=0,
        wavelength_emission=0.532,
        numerical_aperture_detection=1.2,
        index_of_refraction_media=1.3,
    ),
    apply_inverse=fluorescence.ApplyInverseSettings(
        regularization_strength=1e-2,
    ),
)

# API level: simulate + reconstruct
phantom, data = fluorescence.simulate(
    settings,
    recon_dim=2,
    zyx_shape=(100, 256, 256),
    channel_name="GFP",
)
tf = fluorescence.compute_transfer_function(
    data, recon_dim=2, settings=settings
)
result_api = fluorescence.apply_inverse_transfer_function(
    data,
    tf,
    recon_dim=2,
    settings=settings,
    fluor_channel_name="GFP",
)

# Model level: same data, same TF
zyx_data = torch.tensor(data.sel(c="GFP").values, dtype=torch.float32)
singular_system = (
    torch.tensor(tf["singular_system_U"].values),
    torch.tensor(tf["singular_system_S"].values),
    torch.tensor(tf["singular_system_Vh"].values),
)
yx_recon_model = isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
    zyx_data,
    singular_system,
    regularization_strength=1e-2,
)

# Verify model and API match
assert np.allclose(
    result_api.sel(c="GFP_Density2D").values[0],
    yx_recon_model.numpy(),
    atol=1e-6,
)
print("Model-level and API-level outputs match.")

# One-liner
result = fluorescence.reconstruct(
    data,
    recon_dim=2,
    settings=settings,
    fluor_channel_name="GFP",
)
assert np.allclose(result.values, result_api.values)
print("One-liner matches detailed API.")
