"""
API usage examples as runnable tests.

Each test demonstrates a complete reconstruction workflow:
    settings -> input data -> compute TF -> apply inverse -> result

These examples use the API layer directly (no zarr, no CLI).
"""

import numpy as np

from waveorder.api import (
    birefringence,
    birefringence_and_phase,
    fluorescence,
    phase,
)

# --- Phase ---


def test_phase_3d(make_czyx):
    """3D phase reconstruction from brightfield data."""
    settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.532,
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            numerical_aperture_illumination=0.5,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=phase.ApplyInverseSettings(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=1)

    tf = phase.compute_transfer_function(czyx, recon_dim=3, settings=settings)
    result = phase.apply_inverse_transfer_function(
        czyx, tf, recon_dim=3, settings=settings
    )

    assert result.dims == ("c", "z", "y", "x")
    assert list(result.coords["c"].values) == ["Phase3D"]
    assert result.shape == (1, 5, 32, 32)
    assert np.all(np.isfinite(result.values))


def test_phase_2d(make_czyx):
    """2D phase reconstruction (thin sample)."""
    settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.532,
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            z_focus_offset=0,
            numerical_aperture_illumination=0.5,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=phase.ApplyInverseSettings(
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=1)

    tf = phase.compute_transfer_function(czyx, recon_dim=2, settings=settings)
    result = phase.apply_inverse_transfer_function(
        czyx, tf, recon_dim=2, settings=settings
    )

    assert list(result.coords["c"].values) == ["Phase2D"]
    assert result.sizes["z"] == 1


# --- Fluorescence ---


def test_fluorescence_3d(make_czyx):
    """3D fluorescence deconvolution."""
    settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            wavelength_emission=0.507,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=fluorescence.ApplyInverseSettings(
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=1)

    tf = fluorescence.compute_transfer_function(
        czyx, recon_dim=3, settings=settings
    )
    result = fluorescence.apply_inverse_transfer_function(
        czyx,
        tf,
        recon_dim=3,
        settings=settings,
        fluor_channel_name="GFP",
    )

    assert list(result.coords["c"].values) == ["GFP_Density3D"]
    assert result.shape == (1, 5, 32, 32)
    assert np.all(np.isfinite(result.values))


def test_fluorescence_2d(make_czyx):
    """2D fluorescence deconvolution (thin sample)."""
    settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            z_focus_offset=0,
            wavelength_emission=0.507,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=fluorescence.ApplyInverseSettings(
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=1)

    tf = fluorescence.compute_transfer_function(
        czyx, recon_dim=2, settings=settings
    )
    result = fluorescence.apply_inverse_transfer_function(
        czyx,
        tf,
        recon_dim=2,
        settings=settings,
        fluor_channel_name="GFP",
    )

    assert list(result.coords["c"].values) == ["GFP_Density2D"]
    assert result.sizes["z"] == 1


# --- Birefringence ---


def test_birefringence_3d(make_czyx):
    """3D birefringence reconstruction from polarization data."""
    settings = birefringence.Settings(
        transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
        apply_inverse=birefringence.ApplyInverseSettings(
            wavelength_illumination=0.532,
        ),
    )

    czyx = make_czyx(n_channels=4)
    channel_names = [f"ch{i}" for i in range(4)]

    tf = birefringence.compute_transfer_function(czyx, settings, channel_names)
    result = birefringence.apply_inverse_transfer_function(
        czyx, tf, recon_dim=3, settings=settings
    )

    assert list(result.coords["c"].values) == [
        "Retardance",
        "Orientation",
        "Transmittance",
        "Depolarization",
    ]
    assert result.shape[1:] == (5, 32, 32)


# --- Birefringence + Phase ---


def test_birefringence_and_phase_3d(make_czyx):
    """Joint 3D birefringence + phase reconstruction."""
    biref_settings = birefringence.Settings(
        transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
        apply_inverse=birefringence.ApplyInverseSettings(
            wavelength_illumination=0.532,
        ),
    )
    phase_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.532,
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            numerical_aperture_illumination=0.5,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=phase.ApplyInverseSettings(
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=4)
    channel_names = [f"ch{i}" for i in range(4)]

    tf = birefringence_and_phase.compute_transfer_function(
        czyx,
        biref_settings,
        phase_settings,
        channel_names,
        recon_dim=3,
    )
    result = birefringence_and_phase.apply_inverse_transfer_function(
        czyx,
        tf,
        recon_dim=3,
        settings_biref=biref_settings,
        settings_phase=phase_settings,
    )

    expected_channels = [
        "Retardance",
        "Orientation",
        "Transmittance",
        "Depolarization",
        "Phase3D",
        "Retardance_Joint_Decon",
        "Orientation_Joint_Decon",
        "Phase_Joint_Decon",
    ]
    assert list(result.coords["c"].values) == expected_channels
    assert result.shape[1:] == (5, 32, 32)


import pytest


@pytest.mark.xfail(
    reason="Pre-existing bug: 2D biref+phase vector singular system shapes "
    "are incompatible with isotropic_thin_3d.apply_inverse_transfer_function",
    strict=True,
)
def test_birefringence_and_phase_2d(make_czyx):
    """Joint 2D birefringence + phase reconstruction."""
    biref_settings = birefringence.Settings(
        transfer_function=birefringence.TransferFunctionSettings(swing=0.1),
        apply_inverse=birefringence.ApplyInverseSettings(
            wavelength_illumination=0.532,
        ),
    )
    phase_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.532,
            yx_pixel_size=6.5 / 20,
            z_pixel_size=2.0,
            z_focus_offset=0,
            numerical_aperture_illumination=0.5,
            numerical_aperture_detection=1.2,
            index_of_refraction_media=1.3,
        ),
        apply_inverse=phase.ApplyInverseSettings(
            regularization_strength=1e-3,
        ),
    )

    czyx = make_czyx(n_channels=4)
    channel_names = [f"ch{i}" for i in range(4)]

    tf = birefringence_and_phase.compute_transfer_function(
        czyx,
        biref_settings,
        phase_settings,
        channel_names,
        recon_dim=2,
    )
    result = birefringence_and_phase.apply_inverse_transfer_function(
        czyx,
        tf,
        recon_dim=2,
        settings_biref=biref_settings,
        settings_phase=phase_settings,
    )

    expected_channels = [
        "Retardance",
        "Orientation",
        "Transmittance",
        "Depolarization",
        "Phase2D",
    ]
    assert list(result.coords["c"].values) == expected_channels
    assert result.sizes["z"] == 1
