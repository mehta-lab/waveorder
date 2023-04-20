# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import phase3D_3D


v = napari.Viewer()

# 3D OTF parameters
# all lengths must use consistent units e.g. um
tf_args = {
    "ZYX_shape": (100, 256, 256),  # simulation size
    "YX_ps": 6.5 / 63,  # object-space YX pixel size
    "Z_ps": 0.25,  # object-space axial pixel size
    "Z_pad": 0,  # axial padding in pixels
    "lamb_ill": 0.532,  # illumination wavelength
    "n_media": 1.3,  # refractive index in the media
    "NA_ill": 0.9,  # illumination NA
    "NA_obj": 1.2,  # objective NA
}

# Calculate and display OTF
H_re, H_im = phase3D_3D.calc_TF(**tf_args)

ZYX_scale = np.array([tf_args["Z_ps"], tf_args["YX_ps"], tf_args["YX_ps"]])
phase3D_3D.visualize_TF(v, H_re, H_im, ZYX_scale)
input("Showing OTFs. Press <enter> to continue...")
v.layers.select_all()
v.layers.remove_selected()

# Create a phantom
n_sample = 1.50
sphere, _, _ = util.gen_sphere_target(
    tf_args["ZYX_shape"],
    tf_args["YX_ps"],
    tf_args["Z_ps"],
    radius=5,
    blur_size=2 * tf_args["YX_ps"],
)
ZYX_phase = (
    sphere
    * (n_sample - tf_args["n_media"])
    * tf_args["Z_ps"]
    / tf_args["lamb_ill"]
)  # phase in radians

# Perform simulation, reconstruction, and display both
ZYX_data = phase3D_3D.apply_TF(ZYX_phase, H_re, tf_args["Z_pad"])
ZYX_recon = phase3D_3D.apply_inv_TF(
    ZYX_data,
    H_re,
    H_im,
    tf_args["Z_pad"],
    tf_args["Z_ps"],
    tf_args["lamb_ill"],
)

v.add_image(ZYX_phase.numpy(), name="Phantom", scale=ZYX_scale)
v.add_image(ZYX_data.numpy(), name="Data", scale=ZYX_scale)
v.add_image(ZYX_recon.numpy(), name="Reconstruction", scale=ZYX_scale)
input("Showing object, data, and recon. Press <enter> to quit...")
