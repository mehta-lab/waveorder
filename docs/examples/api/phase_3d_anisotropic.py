"""3D phase reconstruction with anisotropic yx pixel sizes.

This example exercises the dict form of ``yx_pixel_size`` for setups where
the y and x pixel sizes differ (e.g. DaXi, light-sheet rigs with asymmetric
binning). The scalar form (``yx_pixel_size=0.1``) is still accepted for
square pixels and routes to the same code path internally.
"""

from waveorder.api import phase

settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        wavelength_illumination=0.532,
        yx_pixel_size={"y": 0.3, "x": 0.25},
        z_pixel_size=0.5,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
        index_of_refraction_media=1.3,
    ),
    apply_inverse=phase.ApplyInverseSettings(
        regularization_strength=1e-3,
    ),
)

phantom, data = phase.simulate(
    settings,
    recon_dim=3,
    zyx_shape=(40, 128, 128),
    index_of_refraction_sample=1.50,
)

result = phase.reconstruct(data, recon_dim=3, settings=settings)

print(f"Output shape: {result.shape}")
print(
    f"yx_pixel_size used: y={settings.transfer_function.yx_pixel_size.y}, "
    f"x={settings.transfer_function.yx_pixel_size.x}"
)
print(f"Channels: {list(result.coords['c'].values)}")
