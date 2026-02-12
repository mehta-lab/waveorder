"""2D phase reconstruction using the waveorder API."""

from waveorder.api import phase

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

# Simulate
phantom, data = phase.simulate(
    settings,
    recon_dim=2,
    zyx_shape=(100, 256, 256),
    index_of_refraction_sample=1.33,
)

# Reconstruct (detailed)
tf = phase.compute_transfer_function(data, recon_dim=2, settings=settings)
result = phase.apply_inverse_transfer_function(
    data, tf, recon_dim=2, settings=settings
)

# Reconstruct (one-liner)
result = phase.reconstruct(data, recon_dim=2, settings=settings)

print(f"Output shape: {result.shape}")
print(f"Channels: {list(result.coords['c'].values)}")
