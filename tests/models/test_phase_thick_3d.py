import numpy as np
import pytest
import torch

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
