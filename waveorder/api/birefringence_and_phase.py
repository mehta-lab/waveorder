"""Joint birefringence + phase reconstruction.

.. warning:: Work in progress. The simulate/reconstruct pipeline works
   end-to-end but the simulation physics and reconstruction quality are
   still being validated. See the birefringence and phase modules
   individually for stable code.
"""

from typing import Literal, Optional

import numpy as np
import torch
import xarray as xr

from waveorder.api import birefringence, phase
from waveorder.api._utils import (
    _biref_inverse_kwargs,
    _build_output_xarray,
    _named_dataarray,
    _output_channel_names,
    _to_singular_system,
    _to_tensor,
    radians_to_nanometers,
)
from waveorder.models import (
    inplane_oriented_thick_pol3d,
    inplane_oriented_thick_pol3d_vector,
    isotropic_fluorescent_thick_3d,
    isotropic_thin_3d,
    phase_thick_3d,
)
from waveorder.stokes import _s12_to_orientation, stokes_after_adr


def simulate(
    settings_biref: birefringence.Settings = None,
    settings_phase: phase.Settings = None,
    zyx_shape: tuple[int, int, int] = (100, 256, 256),
    scheme: str = "4-State",
    index_of_refraction_sample: float = 1.50,
    z_thickness: int = 5,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Simulate polarization data with defocus from a combined phantom.

    Creates a thin star-shaped phantom (z_thickness slices in the center)
    with birefringence and phase properties. The simulation:
    1. Computes sample-plane State intensities from birefringence
    2. Creates a 3D State volume (thin object + background)
    3. Applies widefield defocus (OTF convolution) to each State
    4. Adds WOTF-based phase contrast to S0 for phase reconstruction

    Returns (phantom, data) as CZYX xr.DataArrays.
    Phantom channels: Retardance, Orientation, Transmittance,
                      Depolarization, Phase.
    Data channels: State0, State1, ... (one per polarization state).
    """
    if settings_biref is None:
        settings_biref = birefringence.Settings()
    if settings_phase is None:
        settings_phase = phase.Settings()

    Z, Y, X = zyx_shape
    yx_shape = (Y, X)
    s = settings_phase.transfer_function

    # --- 2D star phantom (birefringence + phase) ---
    retardance, orientation, transmittance, depolarization = inplane_oriented_thick_pol3d.generate_test_phantom(
        yx_shape
    )

    # Phase from same star pattern (in cycles per voxel)
    star = retardance / 0.25  # star normalized to [0, 1]
    delta_n = index_of_refraction_sample - s.index_of_refraction_media
    wavelength_medium = s.wavelength_illumination / s.index_of_refraction_media
    yx_phase = delta_n * s.z_pixel_size / wavelength_medium * star

    # Localize to central z_thickness slices
    z_center = Z // 2
    z_start = z_center - z_thickness // 2
    z_end = z_start + z_thickness

    zyx_phase = torch.zeros((Z, Y, X))
    zyx_phase[z_start:z_end] = yx_phase[None, :, :]

    # --- Birefringence forward: sample-plane State intensities ---
    intensity_to_stokes = inplane_oriented_thick_pol3d.calculate_transfer_function(
        swing=settings_biref.transfer_function.swing,
        scheme=scheme,
    )
    # Sample states (includes +0.1 offset from scalar model)
    cyx_states = inplane_oriented_thick_pol3d.apply_transfer_function(
        retardance,
        orientation,
        transmittance,
        depolarization,
        intensity_to_stokes,
    ).squeeze()  # (C, 1, Y, X) -> (C, Y, X)

    # Background states (no sample: ret=0, ori=0, trans=1, dep=1)
    cyx_bg = inplane_oriented_thick_pol3d.apply_transfer_function(
        torch.zeros(yx_shape),
        torch.zeros(yx_shape),
        torch.ones(yx_shape),
        torch.ones(yx_shape),
        intensity_to_stokes,
    ).squeeze()
    bg_per_state = cyx_bg[:, 0, 0]  # spatially uniform

    # --- Create 3D State volume (thin object + background) ---
    czyx_3d = bg_per_state[:, None, None, None].expand(-1, Z, Y, X).clone()
    num_states = cyx_states.shape[0]
    for c in range(num_states):
        czyx_3d[c, z_start:z_end] = cyx_states[c]

    # --- Apply widefield defocus (OTF convolution) to each State ---
    widefield_otf = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape,
        s.yx_pixel_size,
        s.z_pixel_size,
        wavelength_emission=s.wavelength_illumination,
        z_padding=0,
        index_of_refraction_media=s.index_of_refraction_media,
        numerical_aperture_detection=s.numerical_aperture_detection,
    )

    czyx_defocused = torch.zeros_like(czyx_3d)
    for c in range(num_states):
        czyx_defocused[c] = isotropic_fluorescent_thick_3d.apply_transfer_function(
            czyx_3d[c],
            widefield_otf,
            z_padding=0,
            background=0,
        )

    # --- Add WOTF-based phase contrast to S0 ---
    # The widefield OTF gives defocus blur to all States, but the
    # S0/brightfield also needs WOTF-consistent phase contrast for
    # the phase_thick_3d reconstruction to work.
    stokes_to_intensity = torch.linalg.pinv(intensity_to_stokes)

    # Compute WOTF brightfield
    real_tf, _ = phase_thick_3d.calculate_transfer_function(
        zyx_shape,
        s.yx_pixel_size,
        s.z_pixel_size,
        wavelength_illumination=s.wavelength_illumination,
        z_padding=0,
        index_of_refraction_media=s.index_of_refraction_media,
        numerical_aperture_illumination=s.numerical_aperture_illumination,
        numerical_aperture_detection=s.numerical_aperture_detection,
    )
    # Use the mean S0 as brightness for correct relative contrast
    szyx_stokes = torch.einsum(
        "sc,czyx->szyx",
        intensity_to_stokes,
        czyx_defocused,
    )
    s0_mean = float(szyx_stokes[0].mean())
    zyx_brightfield = phase_thick_3d.apply_transfer_function(
        zyx_phase,
        real_tf,
        z_padding=0,
        brightness=s0_mean,
    )

    # Replace S0 with WOTF brightfield, convert back to States
    szyx_stokes[0] = zyx_brightfield
    czyx_data = torch.einsum(
        "sc,szyx->czyx",
        stokes_to_intensity,
        szyx_stokes,
    )

    # --- Build phantom (all properties localized to central slices) ---
    zyx_coords = {
        "z": np.arange(Z) * s.z_pixel_size,
        "y": np.arange(Y) * s.yx_pixel_size,
        "x": np.arange(X) * s.yx_pixel_size,
    }

    zyx_ret = np.zeros((Z, Y, X), dtype=np.float32)
    zyx_ori = np.zeros((Z, Y, X), dtype=np.float32)
    zyx_trans = np.ones((Z, Y, X), dtype=np.float32)
    zyx_dep = np.ones((Z, Y, X), dtype=np.float32)
    zyx_ret[z_start:z_end] = retardance.numpy()
    zyx_ori[z_start:z_end] = orientation.numpy()
    zyx_trans[z_start:z_end] = transmittance.numpy()
    zyx_dep[z_start:z_end] = depolarization.numpy()

    phantom = xr.DataArray(
        np.stack(
            [
                zyx_ret,
                zyx_ori,
                zyx_trans,
                zyx_dep,
                zyx_phase.numpy(),
            ]
        ),
        dims=("c", "z", "y", "x"),
        coords={
            "c": [
                "Retardance",
                "Orientation",
                "Transmittance",
                "Depolarization",
                "Phase",
            ],
            **zyx_coords,
        },
    )

    num_states = czyx_data.shape[0]
    data = xr.DataArray(
        czyx_data.numpy(),
        dims=("c", "z", "y", "x"),
        coords={
            "c": [f"State{i}" for i in range(num_states)],
            **zyx_coords,
        },
    )
    return phantom, data


def compute_transfer_function(
    czyx_data: xr.DataArray,
    settings_biref: birefringence.Settings,
    settings_phase: phase.Settings,
    input_channel_names: list[str] = None,
    recon_dim: Literal[2, 3] = 3,
) -> xr.Dataset:
    """Compute joint birefringence + phase transfer functions.

    Returns xr.Dataset with:
    - "intensity_to_stokes_matrix": birefringence TF
    - "vector_transfer_function": vector birefringence TF
    - "vector_singular_system_U/S/Vh": vector birefringence singular system
    - For 3D only: "real/imaginary_potential_transfer_function"
    """
    if input_channel_names is None:
        input_channel_names = list(czyx_data.coords["c"].values)

    # Compute birefringence TF
    intensity_to_stokes_matrix = inplane_oriented_thick_pol3d.calculate_transfer_function(
        scheme=str(len(input_channel_names)) + "-State",
        **settings_biref.transfer_function.model_dump(),
    )

    # Compute vector birefringence TF
    zyx_shape = czyx_data.shape[1:]  # CZYX -> ZYX

    num_elements = np.array(zyx_shape).prod()
    max_tf_elements = 1e7
    transverse_downsample_factor = np.ceil(np.sqrt(num_elements / max_tf_elements))

    phase_settings_dict = settings_phase.transfer_function.model_dump()
    phase_settings_dict.pop("z_focus_offset")

    sfZYX_transfer_function, _, singular_system = inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
        zyx_shape=zyx_shape,
        scheme=str(len(input_channel_names)) + "-State",
        **settings_biref.transfer_function.model_dump(),
        **phase_settings_dict,
        fourier_oversample_factor=int(transverse_downsample_factor),
    )

    U, S, Vh = singular_system

    # Build combined Dataset with unique dim names per variable
    variables = {
        "intensity_to_stokes_matrix": _named_dataarray(
            intensity_to_stokes_matrix.cpu().numpy(),
            "intensity_to_stokes_matrix",
        ),
        "vector_transfer_function": _named_dataarray(
            sfZYX_transfer_function.cpu().numpy(),
            "vector_transfer_function",
        ),
        "vector_singular_system_U": _named_dataarray(U.cpu().numpy(), "vector_singular_system_U"),
        "vector_singular_system_S": _named_dataarray(S.cpu().numpy(), "vector_singular_system_S"),
        "vector_singular_system_Vh": _named_dataarray(Vh.cpu().numpy(), "vector_singular_system_Vh"),
    }

    # For 3D, also compute phase TFs (needed by apply_inverse)
    if recon_dim == 3:
        settings_dict = settings_phase.transfer_function.model_dump()
        settings_dict.pop("z_focus_offset")

        real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(zyx_shape=zyx_shape, **settings_dict)

        variables["real_potential_transfer_function"] = _named_dataarray(
            real_tf.cpu().numpy(), "real_potential_transfer_function"
        )
        variables["imaginary_potential_transfer_function"] = _named_dataarray(
            imag_tf.cpu().numpy(), "imaginary_potential_transfer_function"
        )

    return xr.Dataset(variables)


def apply_inverse_transfer_function(
    czyx_data: xr.DataArray,
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3],
    settings_biref: birefringence.Settings,
    settings_phase: phase.Settings,
    cyx_no_sample_data: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Reconstruct joint birefringence and phase.

    2D returns CZYX xr.DataArray with channels
    [Retardance (nm), Orientation, Transmittance, Depolarization, Phase].

    3D returns CZYX xr.DataArray with channels
    [Retardance (nm), Orientation, Transmittance, Depolarization, Phase,
     Retardance_Joint_Decon (nm), Orientation_Joint_Decon, Phase_Joint_Decon].
    """
    wavelength = settings_biref.apply_inverse.wavelength_illumination
    biref_kwargs = _biref_inverse_kwargs(settings_biref)

    czyx_tensor = torch.tensor(czyx_data.values, dtype=torch.float32)
    bg_tensor = torch.tensor(cyx_no_sample_data, dtype=torch.float32) if cyx_no_sample_data is not None else None
    intensity_to_stokes_matrix = _to_tensor(transfer_function, "intensity_to_stokes_matrix")

    # [biref and phase, 2]
    if recon_dim == 2:
        reconstructed_parameters_2d = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
            czyx_tensor,
            intensity_to_stokes_matrix,
            cyx_no_sample_data=bg_tensor,
            project_stokes_to_2d=True,
            **biref_kwargs,
        )

        reconstructed_parameters_3d = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
            czyx_tensor,
            intensity_to_stokes_matrix,
            cyx_no_sample_data=bg_tensor,
            project_stokes_to_2d=False,
            **biref_kwargs,
        )

        brightfield_3d = reconstructed_parameters_3d[2]

        (
            _,
            yx_phase,
        ) = isotropic_thin_3d.apply_inverse_transfer_function(
            brightfield_3d,
            _to_singular_system(transfer_function, "vector_singular_system"),
            **settings_phase.apply_inverse.model_dump(),
        )

        retardance = radians_to_nanometers(reconstructed_parameters_2d[0], wavelength)

        output = torch.stack((retardance,) + reconstructed_parameters_2d[1:] + (torch.unsqueeze(yx_phase, 0),))  # CZYX

    # [biref and phase, 3]
    elif recon_dim == 3:
        reconstructed_parameters_3d = inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
            czyx_tensor,
            intensity_to_stokes_matrix,
            cyx_no_sample_data=bg_tensor,
            project_stokes_to_2d=False,
            **biref_kwargs,
        )

        brightfield_3d = reconstructed_parameters_3d[2]

        zyx_phase = phase_thick_3d.apply_inverse_transfer_function(
            brightfield_3d,
            _to_tensor(transfer_function, "real_potential_transfer_function"),
            _to_tensor(transfer_function, "imaginary_potential_transfer_function"),
            z_padding=settings_phase.transfer_function.z_padding,
            **settings_phase.apply_inverse.model_dump(),
        )

        retardance = radians_to_nanometers(reconstructed_parameters_3d[0], wavelength)

        # Convert retardance and orientation to stokes
        stokes = stokes_after_adr(*reconstructed_parameters_3d)

        stokes = torch.nan_to_num_(torch.stack(stokes), nan=0.0)  # very rare nans from previous line

        joint_recon_params = inplane_oriented_thick_pol3d_vector.apply_inverse_transfer_function(
            szyx_data=stokes,
            singular_system=_to_singular_system(transfer_function, "vector_singular_system"),
            intensity_to_stokes_matrix=None,
            **settings_phase.apply_inverse.model_dump(),
        )

        new_ret = (joint_recon_params[1] ** 2 + joint_recon_params[2] ** 2) ** (0.5)
        new_ori = _s12_to_orientation(joint_recon_params[1], -joint_recon_params[2])

        new_ret_nm = radians_to_nanometers(new_ret, wavelength)

        output = torch.stack(
            (retardance,)
            + reconstructed_parameters_3d[1:]
            + (zyx_phase,)
            + (new_ret_nm,)
            + (new_ori,)
            + (joint_recon_params[0],)
        )

    return _build_output_xarray(
        output.numpy(),
        _output_channel_names(recon_biref=True, recon_phase=True, recon_dim=recon_dim),
        czyx_data,
        singleton_z=(recon_dim == 2),
    )


def reconstruct(
    czyx_data: xr.DataArray,
    settings_biref: birefringence.Settings,
    settings_phase: phase.Settings,
    input_channel_names: list[str] = None,
    recon_dim: Literal[2, 3] = 3,
    cyx_no_sample_data: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Reconstruct joint birefringence and phase (one-liner).

    Chains compute_transfer_function + apply_inverse_transfer_function.
    """
    tf = compute_transfer_function(
        czyx_data,
        settings_biref,
        settings_phase,
        input_channel_names,
        recon_dim,
    )
    return apply_inverse_transfer_function(
        czyx_data,
        tf,
        recon_dim,
        settings_biref,
        settings_phase,
        cyx_no_sample_data,
    )
