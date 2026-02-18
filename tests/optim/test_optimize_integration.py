"""Integration tests for phase and fluorescence optimization examples."""

from waveorder.api import fluorescence, phase
from waveorder.optim import OptimizableFloat


def test_phase_2d_optimize_z_focus_offset():
    """Phase 2D: optimize z_focus_offset from wrong initial guess."""
    gt_offset = 0.6
    gt_settings = phase.Settings(transfer_function=phase.TransferFunctionSettings(z_focus_offset=gt_offset))
    _, data = phase.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
        )
    )
    optimized, recon = phase.optimize(
        data,
        settings=opt_settings,
        num_iterations=50,
        midband_fractions=(0.1, 0.5),
    )

    assert abs(optimized.transfer_function.z_focus_offset - gt_offset) < 0.5


def test_phase_2d_optimize_with_tilt():
    """Phase 2D: optimize z_focus_offset and tilt from wrong initial guess."""
    gt_z, gt_zen = 0.6, 0.5
    gt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=gt_z,
            tilt_angle_zenith=gt_zen,
            tilt_angle_azimuth=1.1,
        )
    )
    _, data = phase.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
            tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.5),
            tilt_angle_azimuth=OptimizableFloat(init=0.0, lr=0.5),
        )
    )
    optimized, recon = phase.optimize(
        data,
        settings=opt_settings,
        num_iterations=50,
        midband_fractions=(0.1, 0.5),
    )

    s = optimized.transfer_function
    assert abs(s.z_focus_offset - gt_z) < 0.5
    assert abs(abs(s.tilt_angle_zenith) - gt_zen) < 0.5


def test_fluorescence_2d_optimize_z_focus_offset():
    """Fluorescence 2D: optimize z_focus_offset from wrong initial guess."""
    gt_offset = 0.6
    gt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(z_focus_offset=gt_offset)
    )
    _, data = fluorescence.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
        )
    )
    optimized, recon = fluorescence.optimize(
        data,
        settings=opt_settings,
        num_iterations=50,
        midband_fractions=(0.1, 0.5),
    )

    assert abs(optimized.transfer_function.z_focus_offset - gt_offset) < 1.0


def test_phase_2d_no_optimizable_params_runs_standard():
    """When no params are optimizable, optimize() falls back to reconstruct()."""
    settings = phase.Settings()
    _, data = phase.simulate(settings, recon_dim=2, zyx_shape=(11, 128, 128))

    returned_settings, recon = phase.optimize(data, settings=settings, num_iterations=10)

    assert returned_settings.transfer_function.z_focus_offset == 0
    assert recon is not None
