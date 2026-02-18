"""Gradient flow tests for modified functions."""

import torch

from waveorder import optics, util
from waveorder.models import isotropic_fluorescent_thin_3d, isotropic_thin_3d


def test_gradient_through_generate_pupil():
    """Gradient flows through smooth generate_pupil."""
    frr = util.generate_radial_frequencies((32, 32), 0.1)
    NA = torch.tensor(0.5, requires_grad=True)
    pupil = optics.generate_pupil(frr, NA, 0.532)
    loss = pupil.sum()
    loss.backward()
    assert NA.grad is not None


def test_gradient_through_tilted_pupil():
    """Gradient flows through tilted pupil."""
    fyy, fxx = util.generate_frequencies((32, 32), 0.1)
    tilt_z = torch.tensor(0.1, requires_grad=True)
    pupil = optics.generate_tilted_pupil(fxx, fyy, 0.5, 0.532, 1.3, tilt_z, 0.0)
    loss = pupil.sum()
    loss.backward()
    assert tilt_z.grad is not None
    assert tilt_z.grad.abs() > 0


def test_gradient_through_propagation_kernel():
    """Gradient flows through clamped propagation kernel."""
    frr = util.generate_radial_frequencies((32, 32), 0.1)
    NA = torch.tensor(0.5, requires_grad=True)
    pupil = optics.generate_pupil(frr, NA, 0.532)
    kernel = optics.generate_propagation_kernel(frr, pupil, 0.532 / 1.3, torch.tensor([-1.0, 0.0, 1.0]))
    loss = kernel.abs().sum()
    loss.backward()
    assert NA.grad is not None


def test_gradient_through_inten_normalization():
    """Gradient flows through functional inten_normalization."""
    data = (torch.rand(3, 32, 32) + 1.0).requires_grad_(True)
    result = util.inten_normalization(data, bg_filter=False)
    # sum(x/mean(x) - 1) == 0 identically, so use squared loss
    loss = (result**2).sum()
    loss.backward()
    assert data.grad is not None
    assert torch.any(data.grad.abs() > 0)


def test_gradient_through_phase_transfer_function():
    """Gradient flows through isotropic_thin_3d TF computation."""
    tilt_z = torch.tensor(0.1, requires_grad=True)
    abs_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(32, 32),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
        tilt_angle_zenith=tilt_z,
        pupil_steepness=100.0,
    )
    loss = abs_tf.abs().sum() + phase_tf.abs().sum()
    loss.backward()
    assert tilt_z.grad is not None
    assert tilt_z.grad.abs() > 0


def test_gradient_through_phase_pseudo_svd():
    """Gradient flows through pseudo-SVD for phase."""
    tilt_z = torch.tensor(0.1, requires_grad=True)
    abs_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(32, 32),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
        tilt_angle_zenith=tilt_z,
        pupil_steepness=100.0,
    )
    U, S, Vh = isotropic_thin_3d.calculate_singular_system(abs_tf, phase_tf, pseudo_svd=True)
    loss = S.sum()
    loss.backward()
    assert tilt_z.grad is not None
    assert tilt_z.grad.abs() > 0


def test_gradient_through_full_phase_pipeline():
    """Gradient flows through full phase reconstruction chain."""
    from waveorder.filter import apply_filter_bank

    tilt_z = torch.tensor(0.1, requires_grad=True)
    abs_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(32, 32),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
        tilt_angle_zenith=tilt_z,
        pupil_steepness=100.0,
    )
    U, S, Vh = isotropic_thin_3d.calculate_singular_system(abs_tf, phase_tf, pseudo_svd=True)
    S_reg = S / (S**2 + 1e-2)
    inverse_filter = torch.einsum("sj...,j...,jf...->fs...", U, S_reg, Vh)

    zyx_data = torch.rand(3, 32, 32) + 10.0
    zyx_norm = util.inten_normalization(zyx_data, bg_filter=False)
    _, phase_yx = apply_filter_bank(inverse_filter, zyx_norm)

    loss = phase_yx.sum()
    loss.backward()
    assert tilt_z.grad is not None


def test_gradient_through_fluorescence_pipeline():
    """Gradient flows through fluorescence TF + pseudo-SVD chain."""
    na_det = torch.tensor(1.2, requires_grad=True)

    fluorescent_tf = isotropic_fluorescent_thin_3d.calculate_transfer_function(
        yx_shape=(32, 32),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_emission=0.507,
        index_of_refraction_media=1.3,
        numerical_aperture_detection=na_det,
    )

    U, S, Vh = isotropic_fluorescent_thin_3d.calculate_singular_system(fluorescent_tf, pseudo_svd=True)
    loss = S.sum()
    loss.backward()
    assert na_det.grad is not None


def test_gradient_through_phase_thick_3d():
    """Gradient flows through 3D phase thick model."""
    from waveorder.filter import apply_filter_bank
    from waveorder.models import phase_thick_3d
    from waveorder.reconstruct import tikhonov_regularized_inverse_filter

    na_det = torch.tensor(1.2, requires_grad=True)
    real_tf, imag_tf = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(5, 16, 16),
        yx_pixel_size=0.1,
        z_pixel_size=0.25,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=na_det,
        pupil_steepness=100.0,
    )
    assert real_tf.requires_grad

    inverse_filter = tikhonov_regularized_inverse_filter(real_tf, 1e-3)
    zyx_data = torch.rand(5, 16, 16) + 10.0
    zyx_norm = util.inten_normalization_3D(zyx_data)
    f_real = apply_filter_bank(inverse_filter[None, None], zyx_norm[None])[0]

    loss = f_real.sum()
    loss.backward()
    assert na_det.grad is not None
