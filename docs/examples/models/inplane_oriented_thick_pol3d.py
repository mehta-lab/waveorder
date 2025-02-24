import napari

from waveorder.models import inplane_oriented_thick_pol3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {"yx_shape": (256, 256)}
transfer_function_arguments = {"swing": 0.1, "scheme": "5-State"}

# Create a phantom
inplane_oriented_parameters = (
    inplane_oriented_thick_pol3d.generate_test_phantom(**simulation_arguments)
)

# Calculate transfer function
intensity_to_stokes_matrix = (
    inplane_oriented_thick_pol3d.calculate_transfer_function(
        **transfer_function_arguments
    )
)

# Display transfer function
viewer = napari.Viewer()
inplane_oriented_thick_pol3d.visualize_transfer_function(
    viewer, intensity_to_stokes_matrix
)
input("Showing transfer functions. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate
czyx_data = inplane_oriented_thick_pol3d.apply_transfer_function(
    *inplane_oriented_parameters,
    intensity_to_stokes_matrix,
)

# Reconstruct
inplane_oriented_parameters_recon = (
    inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
        czyx_data, intensity_to_stokes_matrix
    )
)

# Display
arrays = [
    (inplane_oriented_parameters_recon[3], "Depolarization - recon"),
    (inplane_oriented_parameters_recon[2], "Transmittance - recon"),
    (inplane_oriented_parameters_recon[1], "Orientation (rad) - recon"),
    (inplane_oriented_parameters_recon[0], "Retardance (rad) - recon"),
    (czyx_data, "Data"),
    (inplane_oriented_parameters[3], "Depolarization"),
    (inplane_oriented_parameters[2], "Transmittance"),
    (inplane_oriented_parameters[1], "Orientation (rad)"),
    (inplane_oriented_parameters[0], "Retardance (rad)"),
]

for array in arrays:
    viewer.add_image(array[0].cpu().numpy(), name=array[1])

viewer.grid.enabled = True
viewer.grid.shape = (2, 5)
input("Showing object, data, and recon. Press <enter> to quit...")
