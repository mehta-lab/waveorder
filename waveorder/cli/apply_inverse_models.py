"""
This module converts GUI-level reconstruction calls into library calls
"""

import numpy as np
import torch

from waveorder.models import (
    inplane_oriented_thick_pol3d,
    inplane_oriented_thick_pol3d_vector,
    isotropic_fluorescent_thick_3d,
    isotropic_fluorescent_thin_3d,
    isotropic_thin_3d,
    phase_thick_3d,
)
from waveorder.stokes import _s12_to_orientation, stokes_after_adr


def radians_to_nanometers(retardance_rad, wavelength_illumination_um):
    """
    waveorder returns retardance in radians, while waveorder displays and saves
    retardance in nanometers. This function converts from radians to nanometers
    using the illumination wavelength (which is internally handled in um
    in waveorder).
    """
    return retardance_rad * wavelength_illumination_um * 1e3 / (2 * np.pi)


def birefringence(
    czyx_data,
    cyx_no_sample_data,
    wavelength_illumination,
    recon_dim,
    biref_inverse_dict,
    transfer_function_dataset,
):
    # Load transfer function
    intensity_to_stokes_matrix = torch.tensor(
        transfer_function_dataset["intensity_to_stokes_matrix"][0, 0, 0]
    )

    # Apply reconstruction
    # (retardance, orientation, transmittance, depolarization)
    reconstructed_parameters = (
        inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
            czyx_data,
            intensity_to_stokes_matrix,
            cyx_no_sample_data=cyx_no_sample_data,
            project_stokes_to_2d=(recon_dim == 2),
            **biref_inverse_dict,
        )
    )

    # Convert retardance
    retardance = radians_to_nanometers(
        reconstructed_parameters[0], wavelength_illumination
    )

    return torch.stack((retardance,) + reconstructed_parameters[1:])


def phase(
    czyx_data,
    recon_dim,
    settings_phase,
    transfer_function_dataset,
):
    # [phase only, 2]
    if recon_dim == 2:
        # Load transfer functions
        U = torch.from_numpy(transfer_function_dataset["singular_system_U"][0])
        S = torch.from_numpy(
            transfer_function_dataset["singular_system_S"][0, 0]
        )
        Vh = torch.from_numpy(
            transfer_function_dataset["singular_system_Vh"][0]
        )

        # Apply
        (
            absorption_yx,
            phase_yx,
        ) = isotropic_thin_3d.apply_inverse_transfer_function(
            czyx_data[0],
            (U, S, Vh),
            **settings_phase.apply_inverse.dict(),
        )
        # Stack to C1YX
        output = phase_yx[None, None]
        # TODO: Write phase and absorption to CZYX
        # torch.stack((phase_yx[None], absorption_yx[None]))

    # [phase only, 3]
    elif recon_dim == 3:
        # Load transfer functions
        real_potential_transfer_function = torch.tensor(
            transfer_function_dataset["real_potential_transfer_function"][0, 0]
        )
        imaginary_potential_transfer_function = torch.tensor(
            transfer_function_dataset["imaginary_potential_transfer_function"][
                0, 0
            ]
        )

        # Apply
        output = phase_thick_3d.apply_inverse_transfer_function(
            czyx_data[0],
            real_potential_transfer_function,
            imaginary_potential_transfer_function,
            z_padding=settings_phase.transfer_function.z_padding,
            **settings_phase.apply_inverse.dict(),
        )

    # Pad to CZYX
    while output.ndim != 4:
        output = torch.unsqueeze(output, 0)

    return output


def birefringence_and_phase(
    czyx_data,
    cyx_no_sample_data,
    wavelength_illumination,
    recon_dim,
    biref_inverse_dict,
    settings_phase,
    transfer_function_dataset,
):
    # Load birefringence transfer function
    intensity_to_stokes_matrix = torch.tensor(
        transfer_function_dataset["intensity_to_stokes_matrix"][0, 0, 0]
    )

    # [biref and phase, 2]
    if recon_dim == 2:
        # Load transfer functions
        U = torch.from_numpy(
            transfer_function_dataset["vector_singular_system_U"][0]
        )
        S = torch.from_numpy(
            transfer_function_dataset["vector_singular_system_S"][0, 0]
        )
        Vh = torch.from_numpy(
            transfer_function_dataset["vector_singular_system_Vh"][0]
        )

        # Apply
        reconstructed_parameters_2d = (
            inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
                czyx_data,
                intensity_to_stokes_matrix,
                cyx_no_sample_data=cyx_no_sample_data,
                project_stokes_to_2d=True,
                **biref_inverse_dict,
            )
        )

        reconstructed_parameters_3d = (
            inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
                czyx_data,
                intensity_to_stokes_matrix,
                cyx_no_sample_data=cyx_no_sample_data,
                project_stokes_to_2d=False,
                **biref_inverse_dict,
            )
        )

        brightfield_3d = reconstructed_parameters_3d[2]

        (
            _,
            yx_phase,
        ) = isotropic_thin_3d.apply_inverse_transfer_function(
            brightfield_3d,
            (U, S, Vh),
            **settings_phase.apply_inverse.dict(),
        )

        # Convert retardance
        retardance = radians_to_nanometers(
            reconstructed_parameters_2d[0], wavelength_illumination
        )

        output = torch.stack(
            (retardance,)
            + reconstructed_parameters_2d[1:]
            + (torch.unsqueeze(yx_phase, 0),)
        )  # CZYX

    # [biref and phase, 3]
    elif recon_dim == 3:
        # Load phase transfer functions
        intensity_to_stokes_matrix = torch.tensor(
            transfer_function_dataset["intensity_to_stokes_matrix"][0, 0, 0]
        )
        # Load transfer functions
        real_potential_transfer_function = torch.tensor(
            transfer_function_dataset["real_potential_transfer_function"][0, 0]
        )
        imaginary_potential_transfer_function = torch.tensor(
            transfer_function_dataset["imaginary_potential_transfer_function"][
                0, 0
            ]
        )

        # Apply
        reconstructed_parameters_3d = (
            inplane_oriented_thick_pol3d.apply_inverse_transfer_function(
                czyx_data,
                intensity_to_stokes_matrix,
                cyx_no_sample_data=cyx_no_sample_data,
                project_stokes_to_2d=False,
                **biref_inverse_dict,
            )
        )

        brightfield_3d = reconstructed_parameters_3d[2]

        zyx_phase = phase_thick_3d.apply_inverse_transfer_function(
            brightfield_3d,
            real_potential_transfer_function,
            imaginary_potential_transfer_function,
            z_padding=settings_phase.transfer_function.z_padding,
            **settings_phase.apply_inverse.dict(),
        )

        # Convert retardance
        retardance = radians_to_nanometers(
            reconstructed_parameters_3d[0], wavelength_illumination
        )
        # Load singular system
        U = torch.tensor(
            np.array(transfer_function_dataset["vector_singular_system_U"])
        )
        S = torch.tensor(
            np.array(transfer_function_dataset["vector_singular_system_S"][0])
        )
        Vh = torch.tensor(
            np.array(transfer_function_dataset["vector_singular_system_Vh"])
        )
        singular_system = (U, S, Vh)

        # Convert retardance and orientation to stokes
        stokes = stokes_after_adr(*reconstructed_parameters_3d)

        stokes = torch.nan_to_num_(
            torch.stack(stokes), nan=0.0
        )  # very rare nans from previous like

        # Apply reconstruction
        joint_recon_params = inplane_oriented_thick_pol3d_vector.apply_inverse_transfer_function(
            szyx_data=stokes,
            singular_system=singular_system,
            intensity_to_stokes_matrix=None,
            **settings_phase.apply_inverse.dict(),
        )

        new_ret = (
            joint_recon_params[1] ** 2 + joint_recon_params[2] ** 2
        ) ** (0.5)
        new_ori = _s12_to_orientation(
            joint_recon_params[1], -joint_recon_params[2]
        )

        # Convert stokes to retardance and orientation
        # new_ret, new_ori, _ = estimate_ar_from_stokes012(*joint_recon_params)

        # Convert retardance
        new_ret_nm = radians_to_nanometers(new_ret, wavelength_illumination)

        # Save
        output = torch.stack(
            (retardance,)
            + reconstructed_parameters_3d[1:]
            + (zyx_phase,)
            + (new_ret_nm,)
            + (new_ori,)
            + (joint_recon_params[0],)
        )
    return output


def fluorescence(
    czyx_data, recon_dim, settings_fluorescence, transfer_function_dataset
):
    # [fluo, 2]
    if recon_dim == 2:
        # Load transfer functions for 2D thin fluorescence reconstruction
        U = torch.from_numpy(transfer_function_dataset["singular_system_U"][0])
        S = torch.from_numpy(
            transfer_function_dataset["singular_system_S"][0, 0]
        )
        Vh = torch.from_numpy(
            transfer_function_dataset["singular_system_Vh"][0]
        )

        # Apply 2D fluorescence reconstruction
        output = isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
            czyx_data[0],
            (U, S, Vh),
            **settings_fluorescence.apply_inverse.dict(),
        )
    # [fluo, 3]
    elif recon_dim == 3:
        # Load transfer functions
        optical_transfer_function = torch.tensor(
            transfer_function_dataset["optical_transfer_function"][0, 0]
        )

        # Apply
        output = (
            isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
                czyx_data[0],
                optical_transfer_function,
                settings_fluorescence.transfer_function.z_padding,
                **settings_fluorescence.apply_inverse.dict(),
            )
        )
        # Pad to CZYX
    while output.ndim != 4:
        output = torch.unsqueeze(output, 0)

    return output
