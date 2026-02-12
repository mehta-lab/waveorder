"""3D fluorescence deconvolution using the waveorder API."""

from waveorder.api import fluorescence

settings = fluorescence.Settings(
    transfer_function=fluorescence.TransferFunctionSettings(
        yx_pixel_size=0.1,
        z_pixel_size=0.25,
        wavelength_emission=0.532,
        numerical_aperture_detection=1.2,
        index_of_refraction_media=1.3,
    ),
    apply_inverse=fluorescence.ApplyInverseSettings(
        regularization_strength=1e-3,
    ),
)

# Simulate
phantom, data = fluorescence.simulate(
    settings,
    recon_dim=3,
    zyx_shape=(100, 256, 256),
    channel_name="GFP",
)

# Reconstruct (detailed)
tf = fluorescence.compute_transfer_function(
    data, recon_dim=3, settings=settings
)
result = fluorescence.apply_inverse_transfer_function(
    data,
    tf,
    recon_dim=3,
    settings=settings,
    fluor_channel_name="GFP",
)

# Reconstruct (one-liner)
result = fluorescence.reconstruct(
    data,
    recon_dim=3,
    settings=settings,
    fluor_channel_name="GFP",
)

print(f"Output shape: {result.shape}")
print(f"Channels: {list(result.coords['c'].values)}")
