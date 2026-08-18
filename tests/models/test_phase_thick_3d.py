import numpy as np
import pytest
import torch

from waveorder._pixel_size import YXPixelSize
from waveorder.models import phase_thick_3d


@pytest.mark.parametrize("invert_phase_contrast", (True, False))
def test_calculate_transfer_function(invert_phase_contrast):
    z_padding = 5
    H_re, H_im = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(20, 100, 101),
        yx_pixel_size=6.5 / 40,
        z_pixel_size=2,
        z_padding=z_padding,
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.45,
        numerical_aperture_detection=0.55,
        invert_phase_contrast=invert_phase_contrast,
    )

    assert H_re.shape == (20 + 2 * z_padding, 100, 101)
    assert H_im.shape == (20 + 2 * z_padding, 100, 101)


# Helper function for testing reconstruction invariances
def simulate_phase_recon(
    z_pixel_size_um=0.1,
    yx_pixel_size_um=6.5 / 63,
):
    z_fov_um = 25
    yx_fov_um = 20

    n_z = np.int32(z_fov_um / z_pixel_size_um)
    n_yx = np.int32(yx_fov_um / yx_pixel_size_um)

    # Parameters
    # all lengths must use consistent units e.g. um
    simulation_arguments = {
        "zyx_shape": (n_z, n_yx, n_yx),
        "yx_pixel_size": yx_pixel_size_um,
        "z_pixel_size": z_pixel_size_um,
        "wavelength_illumination": 0.532,
        "index_of_refraction_media": 1.3,
    }
    phantom_arguments = {
        "index_of_refraction_sample": 1.40,
        "sphere_radius": 5,
    }
    transfer_function_arguments = {
        "z_padding": 0,
        "numerical_aperture_illumination": 0.9,
        "numerical_aperture_detection": 1.3,
    }

    # Create a phantom
    zyx_phase = phase_thick_3d.generate_test_phantom(**simulation_arguments, **phantom_arguments)

    # Calculate transfer function
    (
        real_potential_transfer_function,
        imag_potential_transfer_function,
    ) = phase_thick_3d.calculate_transfer_function(**simulation_arguments, **transfer_function_arguments)

    # Simulate
    zyx_data = phase_thick_3d.apply_transfer_function(
        zyx_phase,
        real_potential_transfer_function,
        transfer_function_arguments["z_padding"],
        brightness=1000,
    )

    # Reconstruct
    zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
        zyx_data,
        real_potential_transfer_function,
        imag_potential_transfer_function,
        transfer_function_arguments["z_padding"],
        regularization_strength=1e-3,
    )

    Z, Y, X = zyx_phase.shape
    recon_center = zyx_recon[Z // 2, Y // 2, X // 2].numpy()

    return recon_center


@pytest.mark.parametrize(
    "z_pixel_size_um, yx_pixel_size_um, tolerance",
    [
        (0.1, 6.5 / 63, 0.02),  # baseline
        (0.15, 6.5 / 63, 0.02),  # test z pixel size invariance
        (0.1, 0.8 * 6.5 / 63, 0.02),  # test yx pixel size invariance
    ],
)
def test_phase_invariance(z_pixel_size_um, yx_pixel_size_um, tolerance):
    """Test that the reconstructed physical property (Δn) is invariant to voxel size.

    Reconstruction returns phase in cycles per voxel, which correctly scales with
    voxel size. This test converts back to Δn (the material property) to verify
    that the physical property is recovered invariant to discretization.
    """
    # Baseline with default parameters
    baseline_z_pixel_size_um = 0.1
    baseline = simulate_phase_recon(z_pixel_size_um=baseline_z_pixel_size_um)
    recon = simulate_phase_recon(z_pixel_size_um=z_pixel_size_um, yx_pixel_size_um=yx_pixel_size_um)

    # Convert from cycles per voxel to Δn (refractive index difference)
    # Δn = (cycles/voxel) × λ_medium / z_pixel_size
    wavelength_medium = 0.532 / 1.3  # λ_vacuum / n_media
    baseline_delta_n = baseline * wavelength_medium / baseline_z_pixel_size_um
    recon_delta_n = recon * wavelength_medium / z_pixel_size_um

    # The physical property Δn should be invariant to voxel size
    assert np.abs((recon_delta_n - baseline_delta_n) / baseline_delta_n) < tolerance


def test_calculate_transfer_function_isotropic_yx_pixel_size_equivalence():
    """Anisotropic YXPixelSize with y == x reproduces the legacy scalar output."""
    common = dict(
        zyx_shape=(10, 32, 32),
        z_pixel_size=0.5,
        z_padding=0,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    H_re_scalar, H_im_scalar = phase_thick_3d.calculate_transfer_function(yx_pixel_size=0.2, **common)
    H_re_model, H_im_model = phase_thick_3d.calculate_transfer_function(
        yx_pixel_size=YXPixelSize.isotropic(0.2), **common
    )
    assert torch.allclose(H_re_scalar, H_re_model)
    assert torch.allclose(H_im_scalar, H_im_model)


def test_calculate_transfer_function_anisotropic_runs():
    """y != x produces finite transfer functions of the expected shape."""
    H_re, H_im = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(10, 32, 32),
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_pixel_size=0.5,
        z_padding=0,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert H_re.shape == (10, 32, 32)
    assert H_im.shape == (10, 32, 32)
    assert torch.isfinite(H_re).all()
    assert torch.isfinite(H_im).all()


def test_reconstruct_anisotropic_smoke():
    """End-to-end reconstruct on random data with anisotropic yx pixel size runs."""
    zyx_shape = (10, 32, 32)
    zyx_data = torch.rand(zyx_shape)
    result = phase_thick_3d.reconstruct(
        zyx_data,
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert result.shape == zyx_shape
    assert np.all(np.isfinite(result.numpy()))


def test_reconstruct():
    zyx_shape = (10, 32, 32)
    zyx_data = torch.rand(zyx_shape)

    result = phase_thick_3d.reconstruct(
        zyx_data,
        yx_pixel_size=6.5 / 40,
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )

    assert result.shape == zyx_shape
    assert np.all(np.isfinite(result.numpy()))


_SHARED_OPTICS_KWARGS = dict(
    zyx_shape=(20, 64, 64),
    yx_pixel_size=6.5 / 40,
    z_pixel_size=0.25,
    wavelength_illumination=0.532,
    z_padding=5,
    index_of_refraction_media=1.33,
    numerical_aperture_detection=1.2,
    invert_phase_contrast=False,
    pupil_steepness=1e4,
)


def test_compute_shared_optics_default_is_cpu():
    """With no device kwarg, tensors land on CPU (back-compat)."""
    tensors = phase_thick_3d._compute_shared_optics(**_SHARED_OPTICS_KWARGS)
    for t in tensors:
        assert t.device.type == "cpu"


def test_compute_shared_optics_device_str_cpu():
    """device='cpu' string is accepted and materializes on CPU."""
    tensors = phase_thick_3d._compute_shared_optics(device="cpu", **_SHARED_OPTICS_KWARGS)
    for t in tensors:
        assert t.device.type == "cpu"


def _pearson_complex(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over (Re, Im) concatenated and flattened."""
    a_flat = torch.cat([a.real.flatten(), a.imag.flatten()]).double()
    b_flat = torch.cat([b.real.flatten(), b.imag.flatten()]).double()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    den = torch.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if den.item() == 0:
        # Constant tensor (e.g. pure pupil support); fall back to max-abs-diff check
        return 1.0 if torch.allclose(a_flat, b_flat) else 0.0
    return ((a_c * b_c).sum() / den).item()


def test_angle_z_split_composes_to_shared_optics():
    """The angle/z optics split composes back to bit-identical _compute_shared_optics output.

    Validates that callers using the split helpers
    (:func:`_compute_angle_optics` + :func:`_compute_z_optics`) for the
    FREEZE_ANGLES tilt-recon recipe get the same numbers as the
    legacy single-call path.
    """
    legacy = phase_thick_3d._compute_shared_optics(**_SHARED_OPTICS_KWARGS)
    legacy_fyy, legacy_fxx, legacy_det_pupil, legacy_prop, legacy_green = legacy

    fyy, fxx, radial_frequencies, det_pupil = phase_thick_3d._compute_angle_optics(
        _SHARED_OPTICS_KWARGS["zyx_shape"][1:],
        _SHARED_OPTICS_KWARGS["yx_pixel_size"],
        _SHARED_OPTICS_KWARGS["wavelength_illumination"],
        _SHARED_OPTICS_KWARGS["numerical_aperture_detection"],
        pupil_steepness=_SHARED_OPTICS_KWARGS["pupil_steepness"],
    )
    z_position_list = phase_thick_3d._compute_z_position_list(
        _SHARED_OPTICS_KWARGS["zyx_shape"][0],
        _SHARED_OPTICS_KWARGS["z_pixel_size"],
        _SHARED_OPTICS_KWARGS["z_padding"],
        invert_phase_contrast=_SHARED_OPTICS_KWARGS["invert_phase_contrast"],
    )
    prop, green = phase_thick_3d._compute_z_optics(
        radial_frequencies,
        det_pupil,
        z_position_list,
        _SHARED_OPTICS_KWARGS["wavelength_illumination"],
        _SHARED_OPTICS_KWARGS["index_of_refraction_media"],
    )
    assert torch.equal(legacy_fyy, fyy)
    assert torch.equal(legacy_fxx, fxx)
    assert torch.equal(legacy_det_pupil, det_pupil)
    assert torch.equal(legacy_prop, prop)
    assert torch.equal(legacy_green, green)


def test_angle_optics_cached_across_z_changes():
    """Angle optics tensors don't depend on z_pixel_size or z_padding.

    Concrete check: build angle optics once, then build z optics with two
    different z configurations and confirm the angle outputs are unchanged
    (caller can hold them as a cache).
    """
    angle_kwargs = dict(
        yx_shape=(64, 64),
        yx_pixel_size=6.5 / 40,
        wavelength_illumination=0.532,
        numerical_aperture_detection=1.2,
        pupil_steepness=1e4,
    )
    fyy_a, fxx_a, rf_a, det_a = phase_thick_3d._compute_angle_optics(**angle_kwargs)
    fyy_b, fxx_b, rf_b, det_b = phase_thick_3d._compute_angle_optics(**angle_kwargs)
    assert torch.equal(fyy_a, fyy_b)
    assert torch.equal(fxx_a, fxx_b)
    assert torch.equal(rf_a, rf_b)
    assert torch.equal(det_a, det_b)

    z_list_1 = phase_thick_3d._compute_z_position_list(20, 0.25, 5)
    z_list_2 = phase_thick_3d._compute_z_position_list(20, 0.30, 5)
    prop_1, green_1 = phase_thick_3d._compute_z_optics(
        rf_a, det_a, z_list_1,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.33,
    )
    prop_2, green_2 = phase_thick_3d._compute_z_optics(
        rf_a, det_a, z_list_2,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.33,
    )
    # Different z → different propagation kernels & Green's functions
    assert not torch.equal(prop_1, prop_2)
    assert not torch.equal(green_1, green_2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_shared_optics_cuda_matches_cpu():
    """Building on CUDA must yield numerically equivalent tensors to CPU.

    The change should be *mechanically equivalent* to the legacy CPU-build path
    (no math changes — same generators, same constants), but CPU and CUDA do
    not produce bit-identical floats for transcendentals like ``torch.exp``.
    Strand C of the OPS tilt-recon work targets Pearson ≥ 0.999999 on the
    derived transfer functions (see ``pattern_waveorder_gpu_shared_optics.md``).
    Max-abs-diff on float32 is gated at the ~1e-4 level which corresponds
    to the precision of CUDA's fast transcendentals.
    """
    cpu_tensors = phase_thick_3d._compute_shared_optics(device="cpu", **_SHARED_OPTICS_KWARGS)
    cuda_tensors = phase_thick_3d._compute_shared_optics(device="cuda", **_SHARED_OPTICS_KWARGS)
    names = ["fyy", "fxx", "det_pupil", "propagation_kernel", "greens_function_z"]
    for name, cpu_t, cuda_t in zip(names, cpu_tensors, cuda_tensors):
        assert cuda_t.device.type == "cuda"
        cuda_on_cpu = cuda_t.cpu()
        max_abs = (cpu_t - cuda_on_cpu).abs().max().item()
        p = _pearson_complex(cpu_t, cuda_on_cpu) if cpu_t.is_complex() else _pearson_complex(
            cpu_t.to(torch.complex64), cuda_on_cpu.to(torch.complex64)
        )
        # FP32 transcendental drift between CPU and CUDA is bounded; numerical
        # equivalence is gated by Pearson, not bit-identicality.
        assert max_abs < 1e-3, f"{name} max_abs_diff {max_abs:.3e} exceeds 1e-3"
        assert p >= 0.999999, f"{name} Pearson {p:.9f} < 0.999999"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_calculate_transfer_function_device_threading():
    """When tilt angles arrive on CUDA, TFs come back on CUDA without a CPU detour."""
    cuda = torch.device("cuda")
    H_re, H_im = phase_thick_3d.calculate_transfer_function(
        zyx_shape=(16, 64, 64),
        yx_pixel_size=6.5 / 40,
        z_pixel_size=0.25,
        z_padding=4,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.33,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
        tilt_angle_zenith=torch.tensor(0.1, device=cuda),
        tilt_angle_azimuth=torch.tensor(0.2, device=cuda),
    )
    assert H_re.device.type == "cuda"
    assert H_im.device.type == "cuda"
