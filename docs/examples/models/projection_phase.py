"""
Projection tomography (phase)
==============================

Phase projection tomography with phase transfer function blur via
Siddon ray-tracing. The 3D real-potential TF (weak object approximation)
is extracted as per-angle 2D slices by the Fourier-slice theorem.

Demonstrates both single-shot Tikhonov and iterative CG reconstruction.
"""

import napari
import numpy as np

from waveorder.models import projection_phase

# Parameters
simulation_arguments = {
    "zyx_shape": (64, 64, 64),
    "yx_pixel_size": 0.1,
    "z_pixel_size": 0.1,
}
phantom_arguments = {
    "phantom_type": "point",
    "sphere_radius": 0.25,
}
transfer_function_arguments = {
    "angles": list(range(-70, 75, 5)),
    "wavelength_illumination": 0.532,
    "index_of_refraction_media": 1.3,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
    "z_padding": 0,
    "device": "cpu",
}

# 1. Phantom
phantom = projection_phase.generate_test_phantom(**simulation_arguments, **phantom_arguments)

# 2. Transfer function
siddon_op, otf_slices, real_tf = projection_phase.calculate_transfer_function(
    **simulation_arguments, **transfer_function_arguments
)

# 3. Forward projection (phase TF blur + Siddon)
projections = projection_phase.apply_transfer_function(phantom, siddon_op, real_tf)

# 4. Inverse (Tikhonov single-shot with TF-slice deconvolution)
recon_tik = projection_phase.apply_inverse_transfer_function(
    projections,
    siddon_op,
    otf_slices,
    reconstruction_algorithm="Tikhonov",
    regularization_strength=1e-3,
)

# 5. Inverse (CG iterative with full 3D TF)
recon_cg = projection_phase.apply_inverse_transfer_function(
    projections,
    siddon_op,
    otf_slices,
    reconstruction_algorithm="CG",
    regularization_strength=1e-3,
    n_iter=50,
    real_tf=real_tf,
)

# Display
zyx_scale = np.array(
    [
        simulation_arguments["z_pixel_size"],
        simulation_arguments["yx_pixel_size"],
        simulation_arguments["yx_pixel_size"],
    ]
)

viewer = napari.Viewer()
viewer.add_image(phantom.numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(recon_tik.cpu().numpy(), name="Tikhonov", scale=zyx_scale)
viewer.add_image(recon_cg.cpu().numpy(), name="CG", scale=zyx_scale)
input("Showing phantom and reconstructions. Press <enter> to quit...")
viewer.close()
