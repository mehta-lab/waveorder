import numpy as np
import pytest
import torch

from waveorder.models import isotropic_thin_3d


@pytest.mark.parametrize("invert_phase_contrast", (True, False))
def test_calculate_transfer_function(invert_phase_contrast):
    Hu, Hp = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(100, 101),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1, 0, 1],
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
        invert_phase_contrast=invert_phase_contrast,
    )

    assert Hu.shape == (3, 100, 101)
    assert Hp.shape == (3, 100, 101)


def test_reconstruct():
    yx_shape = (32, 32)
    z_position_list = [-1.0, 0.0, 1.0]
    zyx_data = torch.rand((len(z_position_list),) + yx_shape)

    absorption, phase = isotropic_thin_3d.reconstruct(
        zyx_data,
        yx_pixel_size=6.5 / 40,
        z_position_list=z_position_list,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )

    assert absorption.shape == yx_shape
    assert phase.shape == yx_shape
    assert np.all(np.isfinite(absorption.numpy()))
    assert np.all(np.isfinite(phase.numpy()))


_WRAP_KWARGS = dict(
    yx_shape=(64, 64),
    yx_pixel_size=6.5 / 40,
    z_position_list=[-1.0, 0.0, 1.0],
    wavelength_illumination=0.532,
    index_of_refraction_media=1.33,
    numerical_aperture_illumination=0.4,
    numerical_aperture_detection=0.55,
    invert_phase_contrast=False,
    tilt_angle_zenith=0.1,
    tilt_angle_azimuth=0.2,
    pupil_steepness=1e4,
)


def test_thin_3d_angle_z_split_composes_to_wrap_unsafe():
    """The thin-3d angle/z optics split composes back to bit-identical legacy output."""
    legacy_Hu, legacy_Hp = isotropic_thin_3d._calculate_wrap_unsafe_transfer_function(**_WRAP_KWARGS)

    angle_optics = isotropic_thin_3d._compute_angle_optics(
        _WRAP_KWARGS["yx_shape"],
        _WRAP_KWARGS["yx_pixel_size"],
        _WRAP_KWARGS["wavelength_illumination"],
        _WRAP_KWARGS["index_of_refraction_media"],
        _WRAP_KWARGS["numerical_aperture_illumination"],
        _WRAP_KWARGS["numerical_aperture_detection"],
        tilt_angle_zenith=_WRAP_KWARGS["tilt_angle_zenith"],
        tilt_angle_azimuth=_WRAP_KWARGS["tilt_angle_azimuth"],
        pupil_steepness=_WRAP_KWARGS["pupil_steepness"],
    )
    det_prop = isotropic_thin_3d._compute_z_propagation(
        angle_optics,
        _WRAP_KWARGS["z_position_list"],
        invert_phase_contrast=_WRAP_KWARGS["invert_phase_contrast"],
    )
    Hu, Hp = isotropic_thin_3d._wotf_from_split_optics(angle_optics, det_prop)
    assert torch.equal(legacy_Hu, Hu)
    assert torch.equal(legacy_Hp, Hp)


def test_thin_3d_angle_optics_cached_across_z_changes():
    """Cached angle optics give bit-identical WOTFs when only z changes.

    This is the actual FREEZE_ANGLES workflow: build angle optics ONCE,
    re-call _compute_z_propagation per optimizer iter with new
    z_position_list. Compare against the legacy single-call path
    invoked fresh for each z.
    """
    z_lists = [[-1.0, 0.0, 1.0], [-0.8, 0.0, 0.8], [-1.5, 0.0, 1.5]]
    base = dict(_WRAP_KWARGS)

    angle_optics = isotropic_thin_3d._compute_angle_optics(
        base["yx_shape"],
        base["yx_pixel_size"],
        base["wavelength_illumination"],
        base["index_of_refraction_media"],
        base["numerical_aperture_illumination"],
        base["numerical_aperture_detection"],
        tilt_angle_zenith=base["tilt_angle_zenith"],
        tilt_angle_azimuth=base["tilt_angle_azimuth"],
        pupil_steepness=base["pupil_steepness"],
    )
    for z_list in z_lists:
        kw = {**base, "z_position_list": z_list}
        legacy_Hu, legacy_Hp = isotropic_thin_3d._calculate_wrap_unsafe_transfer_function(**kw)
        det_prop = isotropic_thin_3d._compute_z_propagation(
            angle_optics, z_list, invert_phase_contrast=base["invert_phase_contrast"]
        )
        Hu, Hp = isotropic_thin_3d._wotf_from_split_optics(angle_optics, det_prop)
        assert torch.equal(legacy_Hu, Hu), f"abs TF mismatch at z={z_list}"
        assert torch.equal(legacy_Hp, Hp), f"phase TF mismatch at z={z_list}"


def test_thin_3d_angle_optics_batched_tilt():
    """Batched (B,) tilt angles produce the same split output as legacy."""
    kw = dict(_WRAP_KWARGS)
    kw["tilt_angle_zenith"] = torch.tensor([0.0, 0.1, 0.2])
    kw["tilt_angle_azimuth"] = torch.tensor([0.0, 0.5, 1.0])

    legacy_Hu, legacy_Hp = isotropic_thin_3d._calculate_wrap_unsafe_transfer_function(**kw)
    assert legacy_Hu.shape[0] == 3

    angle_optics = isotropic_thin_3d._compute_angle_optics(
        kw["yx_shape"],
        kw["yx_pixel_size"],
        kw["wavelength_illumination"],
        kw["index_of_refraction_media"],
        kw["numerical_aperture_illumination"],
        kw["numerical_aperture_detection"],
        tilt_angle_zenith=kw["tilt_angle_zenith"],
        tilt_angle_azimuth=kw["tilt_angle_azimuth"],
        pupil_steepness=kw["pupil_steepness"],
    )
    assert angle_optics["batched"]
    det_prop = isotropic_thin_3d._compute_z_propagation(angle_optics, kw["z_position_list"])
    Hu, Hp = isotropic_thin_3d._wotf_from_split_optics(angle_optics, det_prop)
    assert torch.equal(legacy_Hu, Hu)
    assert torch.equal(legacy_Hp, Hp)
