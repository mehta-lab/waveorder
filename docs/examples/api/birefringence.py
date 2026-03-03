"""Birefringence reconstruction using the waveorder API."""

from waveorder.api import birefringence

settings = birefringence.Settings(
    transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
    apply_inverse=birefringence.ApplyInverseSettings(
        wavelength_illumination=0.532,
    ),
)

# Simulate
channel_names = [f"State{i}" for i in range(4)]
phantom, data = birefringence.simulate(
    settings,
    yx_shape=(256, 256),
    scheme="4-State",
)

# Reconstruct (detailed)
tf = birefringence.compute_transfer_function(data, settings, channel_names)
result = birefringence.apply_inverse_transfer_function(data, tf, recon_dim=3, settings=settings)

# Reconstruct (one-liner)
result = birefringence.reconstruct(data, settings, channel_names)

print(f"Output shape: {result.shape}")
print(f"Channels: {list(result.coords['c'].values)}")
