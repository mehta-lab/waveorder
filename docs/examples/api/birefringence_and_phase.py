"""Joint birefringence + phase reconstruction using the waveorder API.

NOTE: Work in progress. The simulate/reconstruct pipeline runs end-to-end
but the simulation physics and reconstruction quality are still being validated.
"""

import numpy as np

from waveorder.api import birefringence, birefringence_and_phase, phase

# Parameters matching the other examples
biref_settings = birefringence.Settings(
    transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
    apply_inverse=birefringence.ApplyInverseSettings(
        wavelength_illumination=0.532,
    ),
)
phase_settings = phase.Settings(
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

# Simulate (star phantom with birefringence + phase, vector forward model)
channel_names = [f"State{i}" for i in range(4)]
phantom, data = birefringence_and_phase.simulate(
    biref_settings,
    phase_settings,
    zyx_shape=(100, 256, 256),
    scheme="4-State",
)
print(f"Phantom channels: {list(phantom.coords['c'].values)}")
print(f"Data channels: {list(data.coords['c'].values)}")

# Reconstruct (detailed)
tf = birefringence_and_phase.compute_transfer_function(
    data, biref_settings, phase_settings, channel_names, recon_dim=3
)
result_detailed = birefringence_and_phase.apply_inverse_transfer_function(
    data,
    tf,
    recon_dim=3,
    settings_biref=biref_settings,
    settings_phase=phase_settings,
)

# Reconstruct (one-liner)
result = birefringence_and_phase.reconstruct(
    data, biref_settings, phase_settings, channel_names, recon_dim=3
)

print(f"Output shape: {result.shape}")
print(f"Channels: {list(result.coords['c'].values)}")

assert np.allclose(result.values, result_detailed.values)
print("Both approaches produce identical results.")
