# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import phase2Dto3D


v = napari.Viewer()

# 3D OTF parameters
# all lengths must use consistent units e.g. um
Z_ps = 0.25 # Z spacing
Z_shape = 100 # Z simulation size
tf_args = {
    "YX_shape": (256, 256),  # simulation size
    "YX_ps": 6.5 / 63,  # object-space YX pixel size
    "Z_pos_list": (np.arange(Z_shape) - Z_shape//2) * Z_ps,
    "lamb_ill": 0.532,  # illumination wavelength
    "n_media": 1.3,  # refractive index in the media
    "NA_ill": 0.9,  # illumination NA
    "NA_obj": 1.2,  # objective NA
}

# Calculate and display OTF
Hu, Hp = phase2Dto3D.calc_TF(**tf_args)

ZYX_scale = np.array([Z_ps, tf_args["YX_ps"], tf_args["YX_ps"]])
phase2Dto3D.visualize_TF(v, Hu, Hp, ZYX_scale)
input("Showing OTFs. Press <enter> to continue...")
v.layers.select_all()
v.layers.remove_selected()

# Create a phantom
n_sample = 1.50
sphere, _, _ = util.gen_sphere_target(
    (Z_shape,) + tf_args["YX_shape"],
    tf_args["YX_ps"],
    Z_ps,
    radius=5,
    blur_size=2 * tf_args["YX_ps"],
)
ZYX_phase = (
    sphere
    * (n_sample - tf_args["n_media"])
    * Z_ps
    / tf_args["lamb_ill"]
)  # phase in radians

# Perform simulation, reconstruction, and display both
ZYX_data = phase2Dto3D.apply_TF(ZYX_phase, Hp)
ZYX_recon = phase2Dto3D.apply_inv_TF(
    ZYX_data, Hu, Hp
)

v.add_image(ZYX_phase.numpy(), name="Phantom", scale=ZYX_scale)
v.add_image(ZYX_data.numpy(), name="Data", scale=ZYX_scale)
v.add_image(ZYX_recon.numpy(), name="Reconstruction", scale=ZYX_scale[1:])
input("Showing object, data, and recon. Press <enter> to quit...")
