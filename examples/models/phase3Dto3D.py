# 3D partially coherent optical diffraction tomography (ODT) simulation                                                                  #
# ```J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative 3D tomographic imaging for partially    #
# coherent light microscopy," Opt. Express 25, 15699-15712 (2017).```                                           #

import torch
import napari
import numpy as np
from waveorder import util
from waveorder.models import phase3Dto3D

v = napari.Viewer()

# Experiment parameters
tf_params = {
    "ZYX_shape": (100, 256, 256),  # simulation size
    "YX_ps": 6.5 / 63,  # effective pixel size
    "Z_ps": 0.25,  # axial pixel size
    "Z_pad": 0,  # paddding
    "lamb_ill": 0.532,  # wavelength
    "n_media": 1.3,  # refractive index in the media
    "NA_ill": 0.9,  # illumination NA
    "NA_obj": 1.2,  # objective NA
}

# Generate a phantom
n_sample = 1.50
radius = 5
blur_size = 2 * tf_params["YX_ps"]
sphere, _, _ = util.gen_sphere_target(
    tf_params["ZYX_shape"],
    tf_params["YX_ps"],
    tf_params["Z_ps"],
    radius,
    blur_size,
)
RI_map = torch.zeros_like(sphere)
RI_map[sphere > 0] = sphere[sphere > 0] * (n_sample - tf_params["n_media"])
RI_map += tf_params["n_media"]
t_obj = torch.exp(
    1j * 2 * np.pi * tf_params["Z_ps"] * (RI_map - tf_params["n_media"])
)

# Display input bead
ZYX_scale = np.array(
    [tf_params["Z_ps"], tf_params["YX_ps"], tf_params["YX_ps"]]
)
v.add_image(torch.angle(t_obj).numpy(), name="Phantom", scale=ZYX_scale)
input("Showing input phantom. Press <enter> to continue...")
v.layers.remove("Phantom")

# Calculate and display OTF
H_re, H_im = phase3Dto3D.calc_TF(**tf_params)
phase3Dto3D.visualize_TF(v, H_re, H_im, ZYX_scale)
input("Showing OTFs. Press <enter> to continue...")
v.layers.remove("Re(H_im)")
v.layers.remove("Im(H_im)")
v.layers.remove("Re(H_re)")
v.layers.remove("Im(H_re)")

# Perform simulation, reconstruction, and display both
ZYX_data = phase3Dto3D.apply_TF(t_obj, H_re)
ZYX_recon = phase3Dto3D.apply_inv_TF(
    ZYX_data, H_re, H_im, tf_params["Z_ps"], tf_params["lamb_ill"]
)

v.add_image(torch.angle(t_obj).numpy(), name="Phantom", scale=ZYX_scale)
v.add_image(ZYX_data.numpy(), name="Data", scale=ZYX_scale)
v.add_image(ZYX_recon.numpy(), name="Reconstruction", scale=ZYX_scale)
input("Showing object, data, and recon. Press <enter> to quit...")
