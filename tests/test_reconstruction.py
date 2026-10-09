"""Values, derivatives, ownership and lifecycle of the opt-in reconstruction interface."""

import math
import os

import pytest
import torch

from waveorder.models import phase_thick_3d
from waveorder.reconstruction import PhaseReconstruction


_SETTINGS = dict(
    yx_pixel_size=0.22, z_pixel_size=0.65, wavelength_illumination=0.532,
    z_padding=2, index_of_refraction_media=1.3,
    numerical_aperture_illumination=0.4, numerical_aperture_detection=0.7,
    invert_phase_contrast=False, tilt_angle_zenith=0.14, tilt_angle_azimuth=0.31,
    pupil_steepness=100.0, regularization_strength=0.02, absorption_ratio=0.15,
)


def _assert_gradient_close(actual, reference):
    scale = reference.abs().max().item()
    torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-5 * scale)
    reference_norm = torch.linalg.vector_norm(reference)
    error_norm = torch.linalg.vector_norm(actual - reference)
    if reference_norm == 0:
        assert error_norm == 0
    else:
        assert error_norm / reference_norm <= 3e-3


@pytest.fixture(params=[
    ("torch", "cpu"),
    pytest.param(("torch", "cuda:0"), marks=pytest.mark.gpu),
    pytest.param(("cuda", "cuda:0"), marks=pytest.mark.gpu),
])
def execution(request):
    backend, device = request.param
    if device.startswith("cuda"):
        if os.environ.get("WAVEORDER_TEST_CUDA") != "1":
            pytest.skip("Set WAVEORDER_TEST_CUDA=1 to exercise GPU reconstruction")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
    return backend, device


def _data(shape):
    return 1 + 0.2 * torch.rand(shape, generator=torch.Generator().manual_seed(41))


def _reference(data, settings):
    settings = dict(settings)
    if settings["absorption_ratio"] is None:
        settings["absorption_ratio"] = 0.0
    return phase_thick_3d.reconstruct(data, **settings)


@pytest.mark.parametrize(("shape", "padding"), [
    ((5, 7, 9), 0), ((6, 8, 10), 2), ((5, 8, 9), 5), ((6, 7, 10), 7),
])
def test_full_domain_values_and_stable_outputs(execution, shape, padding):
    backend, device = execution
    settings = dict(_SETTINGS, z_padding=padding)
    data = _data(shape)
    expected = _reference(data, settings)
    changed = data.flip(0).contiguous()
    expected_changed = _reference(changed, settings)
    with PhaseReconstruction(shape, **settings, backend=backend, device=device) as reconstruction:
        first = reconstruction(data.to(device))
        second = reconstruction(changed.to(device))
        destination = torch.empty(shape, device=device)
        assert reconstruction(data.to(device), out=destination) is destination
        torch.testing.assert_close(first.cpu(), expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(second.cpu(), expected_changed, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(destination.cpu(), expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(first.cpu(), expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(second.cpu(), expected_changed, rtol=2e-4, atol=2e-4)
    reconstruction.close()
    with pytest.raises(RuntimeError):
        reconstruction(data.to(device))


@pytest.mark.parametrize("shape", [(5, 7, 9), (6, 8, 10), (5, 8, 10)])
def test_compact_coefficients_match_full_transfer_projection(shape):
    from waveorder.reconstruction import _torch
    from waveorder._pixel_size import YXPixelSize

    settings = dict(_SETTINGS)
    optical = {key: value for key, value in settings.items()
               if key not in ("regularization_strength", "absorption_ratio")}
    real, imaginary = phase_thick_3d.calculate_transfer_function(shape, **optical)
    transfer = real + settings["absorption_ratio"] * imaginary
    inverse = transfer.conj() / (transfer.conj() * transfer + settings["regularization_strength"])
    negative = torch.roll(torch.flip(inverse, (-3, -2, -1)), (1, 1, 1), (-3, -2, -1))
    expected = ((inverse + negative.conj()) * 0.5)[..., :shape[-1] // 2 + 1]
    settings["yx_pixel_size"] = YXPixelSize.from_value(settings["yx_pixel_size"])
    actual = _torch.prepare_filter(shape, **settings, device=torch.device("cpu"))
    torch.testing.assert_close(actual.values, expected, rtol=1e-4, atol=3e-6 * expected.abs().max().item() + 1e-12)
    relative_l2 = torch.linalg.vector_norm(actual.values - expected) / torch.linalg.vector_norm(expected)
    assert relative_l2 < 3e-5


def test_tensor_parameter_updates_build_fresh_optical_and_filter_graphs():
    shape = (5, 8, 9)
    names = ("numerical_aperture_illumination", "numerical_aperture_detection",
             "tilt_angle_zenith", "tilt_angle_azimuth", "regularization_strength", "absorption_ratio")
    settings = dict(_SETTINGS)
    parameters = {name: torch.tensor(0.0 if name == "absorption_ratio" else settings[name], requires_grad=True)
                  for name in names}
    settings.update(parameters)
    probe = torch.rand(shape, generator=torch.Generator().manual_seed(53))
    with PhaseReconstruction(shape, **settings) as reconstruction:
        for detection_na, ratio in ((0.7, 0.0), (1.17, 0.1), (0.72, 0.0)):
            with torch.no_grad():
                parameters["numerical_aperture_detection"].fill_(detection_na)
                parameters["absorption_ratio"].fill_(ratio)
            data = _data(shape).requires_grad_()
            reference_data = data.detach().clone().requires_grad_()
            reference_parameters = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
            actual = reconstruction(data)
            expected = _reference(reference_data, dict(_SETTINGS, **reference_parameters))
            loss = (actual * probe).mean() + actual.square().mean()
            reference_loss = (expected * probe).mean() + expected.square().mean()
            gradients = torch.autograd.grad(loss, (data, *parameters.values()))
            reference_gradients = torch.autograd.grad(reference_loss, (reference_data, *reference_parameters.values()))
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            for observed, reference in zip(gradients, reference_gradients):
                _assert_gradient_close(observed, reference)
            assert reference_gradients[-1].abs() > 1e-8


@pytest.mark.parametrize("parameter", ["regularization_strength", "absorption_ratio"])
def test_filter_only_gradients_with_fixed_optics(parameter):
    shape = (5, 7, 9)
    leaf = torch.tensor(0.0 if parameter == "absorption_ratio" else 0.02, requires_grad=True)
    settings = dict(_SETTINGS, **{parameter: leaf})
    with PhaseReconstruction(shape, **settings) as reconstruction:
        for _ in range(2):
            actual = reconstruction(_data(shape))
            reference_leaf = leaf.detach().clone().requires_grad_()
            expected = _reference(_data(shape), dict(_SETTINGS, **{parameter: reference_leaf}))
            gradient, = torch.autograd.grad(actual.square().mean(), leaf)
            reference, = torch.autograd.grad(expected.square().mean(), reference_leaf)
            _assert_gradient_close(gradient, reference)
            assert reference.abs() > 1e-8


def test_cached_scalar_filter_keeps_repeated_input_gradients():
    shape = (5, 7, 9)
    with torch.inference_mode():
        reconstruction = PhaseReconstruction(shape, **_SETTINGS)
    with reconstruction:
        for _ in range(2):
            data = _data(shape).requires_grad_()
            reference_data = data.detach().clone().requires_grad_()
            gradient, = torch.autograd.grad(reconstruction(data).square().sum(), data)
            reference, = torch.autograd.grad(_reference(reference_data, _SETTINGS).square().sum(), reference_data)
            _assert_gradient_close(gradient, reference)


@pytest.mark.parametrize("regularization", [-1e-50, float("nan"), float("inf"), 0.01j,
                                           torch.tensor(math.nextafter(float(torch.finfo(torch.float32).max), math.inf), dtype=torch.float64)])
def test_invalid_regularization_rejected_before_narrowing(regularization):
    with pytest.raises(ValueError):
        PhaseReconstruction((5, 7, 9), **dict(_SETTINGS, regularization_strength=regularization))


def test_zero_regularization_keeps_reference_singularities():
    data = _data((5, 8, 8))
    settings = dict(_SETTINGS, regularization_strength=0.0, absorption_ratio=None,
                    z_padding=0, yx_pixel_size=0.05, tilt_angle_zenith=0.0, tilt_angle_azimuth=0.0)
    expected = _reference(data, settings)
    assert not torch.isfinite(expected).all()
    with PhaseReconstruction(data.shape, **settings) as reconstruction:
        torch.testing.assert_close(reconstruction(data), expected, equal_nan=True)


def test_output_and_input_errors_leave_operator_usable(execution):
    backend, device = execution
    shape = (5, 7, 9)
    data = _data(shape).to(device)
    with PhaseReconstruction(shape, **_SETTINGS, backend=backend, device=device) as reconstruction:
        for invalid in (data.double(), data.half(), data.unsqueeze(0), data.transpose(1, 2)):
            with pytest.raises(ValueError):
                reconstruction(invalid)
        with pytest.raises(ValueError):
            reconstruction(data, out=data)
        with pytest.raises(ValueError):
            reconstruction(data, out=torch.empty_like(data, requires_grad=True))
        with pytest.raises(ValueError):
            reconstruction(data.detach().requires_grad_(), out=torch.empty_like(data))
        if backend == "cuda":
            with pytest.raises(ValueError):
                reconstruction(data.detach().requires_grad_())
        torch.testing.assert_close(reconstruction(data).cpu(), _reference(data.cpu(), _SETTINGS), rtol=2e-4, atol=2e-4)


def test_native_rejects_parameter_gradients_before_loading():
    with pytest.raises(ValueError):
        PhaseReconstruction((5, 7, 9), **dict(_SETTINGS, absorption_ratio=torch.tensor(0.0, requires_grad=True)),
                            backend="cuda", device="cuda")


@pytest.mark.parametrize("pixels", [
    torch.tensor(0.22, requires_grad=True),
    {"y": torch.tensor(0.22, requires_grad=True), "x": 0.22},
])
def test_fixed_sampling_rejects_tensors_before_scalar_conversion(pixels):
    with pytest.raises(ValueError):
        PhaseReconstruction((5, 7, 9), **dict(_SETTINGS, yx_pixel_size=pixels))


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_private_filter_readiness_cross_stream_and_close(backend):
    if os.environ.get("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in CUDA test")
    shape = (5, 7, 9)
    source = _data(shape)
    expected = _reference(source, _SETTINGS)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        data = source.cuda()
        input_ready = torch.cuda.Event()
        input_ready.record()
        torch.cuda._sleep(2_000_000)
        reconstruction = PhaseReconstruction(shape, **_SETTINGS, backend=backend, device="cuda")
    with torch.cuda.stream(consumer):
        consumer.wait_event(input_ready)  # Does not cover the later private filter preparation.
        output = reconstruction(data)
    del data
    reconstruction.close()
    torch.testing.assert_close(output.cpu(), expected, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
def test_native_snapshots_tensor_parameters():
    if os.environ.get("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in CUDA test")
    shape = (5, 7, 9)
    na = torch.tensor(0.7)
    with PhaseReconstruction(shape, **dict(_SETTINGS, numerical_aperture_detection=na),
                             backend="cuda", device="cuda") as reconstruction:
        na.fill_(1.17)
        actual = reconstruction(_data(shape).cuda())
    torch.testing.assert_close(actual.cpu(), _reference(_data(shape), _SETTINGS), rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_cross_stream_input_and_output_allocator_lifetimes(backend):
    if os.environ.get("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in CUDA test")
    shape = (9, 14, 17)
    source = _data(shape)
    expected = _reference(source, _SETTINGS)
    allocator_stream = torch.cuda.Stream()
    execution_stream = torch.cuda.Stream()
    with PhaseReconstruction(shape, **_SETTINGS, backend=backend, device="cuda") as reconstruction:
        with torch.cuda.stream(allocator_stream):
            data = source.cuda()
            destination = torch.empty_like(data)
            ready = torch.cuda.Event()
            ready.record()
        with torch.cuda.stream(execution_stream):
            execution_stream.wait_event(ready)
            torch.cuda._sleep(5_000_000)
            saved = reconstruction(data, out=destination).clone()
        del data, destination
        # Reuse the allocation stream before execution completes. Without
        # record_stream on both external buffers these allocations can corrupt it.
        with torch.cuda.stream(allocator_stream):
            replacements = [torch.empty(shape, device="cuda").fill_(99) for _ in range(16)]
        execution_stream.synchronize()
        torch.testing.assert_close(saved.cpu(), expected, rtol=2e-4, atol=2e-4)
        del replacements


@pytest.mark.gpu
def test_native_workspace_serializes_calls_on_different_streams():
    if os.environ.get("WAVEORDER_TEST_CUDA") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt-in CUDA test")
    shape = (9, 14, 17)
    source = _data(shape)
    changed = source.flip(0).contiguous()
    data, other = source.cuda(), changed.cuda()
    ready = torch.cuda.Event()
    ready.record()
    first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with PhaseReconstruction(shape, **_SETTINGS, backend="cuda", device="cuda") as reconstruction:
        with torch.cuda.stream(first_stream):
            first_stream.wait_event(ready)
            torch.cuda._sleep(5_000_000)
            first = reconstruction(data)
        with torch.cuda.stream(second_stream):
            second_stream.wait_event(ready)
            second = reconstruction(other)
    torch.testing.assert_close(first.cpu(), _reference(source, _SETTINGS), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(second.cpu(), _reference(changed, _SETTINGS), rtol=2e-4, atol=2e-4)
