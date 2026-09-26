import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from waveorder._pixel_size import YXPixelSize
from waveorder.api import fluorescence as fluorescence_api
from waveorder.cli import settings
from waveorder.cli.apply_inverse_transfer_function import (
    _warn_pixel_size_mismatch,
    apply_inverse_transfer_function_cli,
)
from waveorder.cli.main import cli
from waveorder.io import utils

input_scale = [1, 2, 3, 4, 5]
# Setup options
birefringence_settings = settings.BirefringenceSettings()

# birefringence_option, time_indices, phase_option, dimension_option, time_length_target
all_options = [
    (birefringence_settings, [0, 3, 4], None, 2, 5),
    (birefringence_settings, 0, settings.PhaseSettings(), 2, 5),
    (birefringence_settings, [0, 1], None, 3, 5),
    (birefringence_settings, "all", settings.PhaseSettings(), 3, 5),
]


@pytest.fixture(scope="session")
def tmp_input_path_zarr():
    tmp_path = TemporaryDirectory()
    yield Path(tmp_path.name) / "input.zarr", Path(tmp_path.name) / "test.yml"
    tmp_path.cleanup()


def test_reconstruct(tmp_input_path_zarr):
    input_path, tmp_config_yml = tmp_input_path_zarr
    # Generate input "dataset"
    channel_names = [f"State{x}" for x in range(4)]
    dataset = open_ome_zarr(
        input_path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )

    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (5, 4, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=input_scale)],
    )

    for i, (
        birefringence_option,
        time_indices,
        phase_option,
        dimension_option,
        time_length_target,
    ) in enumerate(all_options):
        if (birefringence_option is None) and (phase_option is None):
            continue

        # Generate recon settings
        recon_settings = settings.ReconstructionSettings(
            input_channel_names=channel_names,
            time_indices=time_indices,
            reconstruction_dimension=dimension_option,
            birefringence=birefringence_option,
            phase=phase_option,
        )
        config_path = tmp_config_yml.with_name(f"{i}.yml")
        utils.model_to_yaml(recon_settings, config_path)

        # Run CLI
        runner = CliRunner()
        tf_path = input_path.with_name(f"tf_{i}.zarr")
        runner.invoke(
            cli,
            [
                "compute-tf",
                "-i",
                str(input_path / "0" / "0" / "0"),
                "-c",
                str(config_path),
                "-o",
                str(tf_path),
            ],
            catch_exceptions=False,
        )
        assert tf_path.exists()


def test_append_channel_reconstruction(tmp_input_path_zarr):
    input_path, tmp_config_yml = tmp_input_path_zarr
    output_path = input_path.with_name("output.zarr")

    # Generate input "dataset"
    channel_names = [f"State{x}" for x in range(4)] + ["GFP"]
    dataset = open_ome_zarr(
        input_path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (5, 5, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=input_scale)],
    )

    # Generate recon settings
    biref_settings = settings.ReconstructionSettings(
        input_channel_names=[f"State{x}" for x in range(4)],
        time_indices="all",
        reconstruction_dimension=3,
        birefringence=settings.BirefringenceSettings(),
        phase=None,
        fluorescence=None,
    )
    fluor_settings = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        time_indices="all",
        reconstruction_dimension=3,
        birefringence=None,
        phase=None,
        fluorescence=settings.FluorescenceSettings(),
    )
    biref_config_path = tmp_config_yml.with_name("biref.yml")
    fluor_config_path = tmp_config_yml.with_name("fluor.yml")

    utils.model_to_yaml(biref_settings, biref_config_path)
    utils.model_to_yaml(fluor_settings, fluor_config_path)

    # Apply birefringence reconstruction

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(biref_config_path),
            "-o",
            str(output_path),
            "-uid",
            str("birefringence_reconstruction"),
        ],
        catch_exceptions=False,
    )

    assert output_path.exists()
    with open_ome_zarr(output_path) as dataset:
        assert dataset["0/0/0"]["0"].shape[1] == 4

    # Append fluorescence reconstruction
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(fluor_config_path),
            "-o",
            str(output_path),
            "-uid",
            str("fluorescence_reconstruction"),
        ],
        catch_exceptions=False,
    )

    assert output_path.exists()

    with open_ome_zarr(output_path) as dataset:
        assert dataset["0/0/0"]["0"].shape[1] == 5
        assert dataset.channel_names[-1] == "GFP_Density3D"
        assert dataset.channel_names[-2] == "Depolarization"

        # Each reconstruction keeps its own top-level provenance key (#206)
        position = dataset["0/0/0"]
        assert position.zattrs["waveorder-Birefringence"] == biref_settings.model_dump()
        assert position.zattrs["waveorder-GFP_Density3D"] == fluor_settings.model_dump()
        assert "waveorder" not in position.zattrs


def test_fluorescence_2d_reconstruction(tmp_input_path_zarr):
    """Test 2D fluorescence reconstruction through CLI"""
    input_path, tmp_config_yml = tmp_input_path_zarr
    output_path = input_path.with_name("fluor_2d_output.zarr")

    # Generate input "dataset" with fluorescence channel
    channel_names = [f"State{x}" for x in range(4)] + ["GFP"]
    dataset = open_ome_zarr(
        input_path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (5, 5, 4, 5, 6),  # T, C, Z, Y, X - one fluorescence channel
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=input_scale)],
    )

    # Generate 2D fluorescence reconstruction settings
    fluor_2d_settings = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        time_indices="all",
        reconstruction_dimension=2,  # 2D reconstruction
        birefringence=None,
        phase=None,
        fluorescence=settings.FluorescenceSettings(),
    )
    fluor_2d_config_path = tmp_config_yml.with_name("fluor_2d.yml")
    utils.model_to_yaml(fluor_2d_settings, fluor_2d_config_path)

    # Run 2D fluorescence reconstruction
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(fluor_2d_config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )
    assert output_path.exists()

    # Verify output structure
    with open_ome_zarr(output_path) as dataset:
        # Should have 1 channel for 2D fluorescence density
        assert dataset["0/0/0"]["0"].shape[1] == 1
        # Should have Z=1 for 2D reconstruction
        assert dataset["0/0/0"]["0"].shape[2] == 1
        # YX dimensions should match input
        assert dataset["0/0/0"]["0"].shape[3:] == (5, 6)
        assert dataset.channel_names[-1] == "GFP_Density2D"


def test_optimization_cli(tmp_path):
    """Smoke test: reconstruct with OptimizableFloat runs optimization then reconstructs."""
    from waveorder.api.phase import TransferFunctionSettings as PhaseTFSettings
    from waveorder.cli.settings import OptimizationSettings

    input_path = tmp_path / "optim_input.zarr"
    output_path = tmp_path / "optim_output.zarr"

    # Create input dataset with a single channel
    channel_names = ["Brightfield"]
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    data = np.random.rand(1, 1, 4, 32, 32).astype(np.float32) + 10.0
    position.create_image("0", data, transform=[TransformationMeta(type="scale", scale=input_scale)])
    dataset.close()

    # Config with an optimizable z_focus_offset
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=channel_names,
        time_indices=0,
        reconstruction_dimension=2,
        phase=settings.PhaseSettings(
            transfer_function=PhaseTFSettings(z_focus_offset={"init": 0, "lr": 0.1}),
        ),
        optimization=OptimizationSettings(max_iterations=2),
    )
    config_path = tmp_path / "optim.yml"
    utils.model_to_yaml(recon_settings, config_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert output_path.exists()

    # Verify optimized config was saved with updated parameters
    optimized_config = config_path.with_name("optim_optimized.yml")
    assert optimized_config.exists()
    optimized = utils.yaml_to_model(optimized_config, settings.ReconstructionSettings)
    # z_focus_offset should be a plain float (no longer OptimizableFloat) and should have moved from 0
    assert isinstance(optimized.phase.transfer_function.z_focus_offset, float)
    assert optimized.phase.transfer_function.z_focus_offset != 0.0


def test_cli_apply_inv_tf_mock(tmp_input_path_zarr):
    tmp_input_zarr, tmp_config_yml = tmp_input_path_zarr
    tmp_config_yml = tmp_config_yml.with_name("0.yml").resolve()
    tf_path = tmp_input_zarr.with_name("tf_0.zarr").resolve()
    input_path = (tmp_input_zarr / "0" / "0" / "0").resolve()
    result_path = tmp_input_zarr.with_name("result.zarr").resolve()

    assert tmp_config_yml.exists()
    assert tf_path.exists()
    assert input_path.exists()
    assert not result_path.exists()

    runner = CliRunner()
    with patch("waveorder.cli.apply_inverse_transfer_function.apply_inverse_transfer_function_cli") as mock:
        cmd = [
            "apply-inv-tf",
            "-i",
            str(input_path),
            "-t",
            str(tf_path),
            "-c",
            str(tmp_config_yml),
            "-o",
            str(result_path),
            "-j",
            str(1),
        ]
        result_inv = runner.invoke(
            cli,
            cmd,
            catch_exceptions=False,
        )
        mock.assert_called_with(
            [input_path],
            Path(tf_path),
            Path(tmp_config_yml),
            Path(result_path),
            1,
            False,
            False,
        )
        assert result_inv.exit_code == 0


def test_cli_apply_inv_tf_output(tmp_input_path_zarr, capsys):
    tmp_input_zarr, tmp_config_yml = tmp_input_path_zarr
    input_path = tmp_input_zarr / "0" / "0" / "0"

    for i, (
        birefringence_option,
        time_indices,
        phase_option,
        dimension_option,
        time_length_target,
    ) in enumerate(all_options):
        if (birefringence_option is None) and (phase_option is None):
            continue

        result_path = tmp_input_zarr.with_name(f"result{i}.zarr").resolve()

        tf_path = tmp_input_zarr.with_name(f"tf_{i}.zarr")
        tmp_config_yml = tmp_config_yml.with_name(f"{i}.yml")

        # # Check output
        apply_inverse_transfer_function_cli([input_path], tf_path, tmp_config_yml, result_path, 1)

        result_dataset = open_ome_zarr(str(result_path / "0" / "0" / "0"))
        assert result_dataset["0"].shape[0] == time_length_target
        assert result_dataset["0"].shape[3:] == (5, 6)

        assert result_path.exists()
        captured = capsys.readouterr()
        assert "Starting reconstruction" in captured.out

        # Check scale transformations pass through
        assert input_scale == result_dataset.scale


def test_pixel_size_mismatch_warning(tmp_path):
    """Warn when input zarr scale and config pixel sizes differ by >5%."""
    import warnings as _warnings

    input_path = tmp_path / "mismatch_input.zarr"
    output_path = tmp_path / "mismatch_output.zarr"

    channel_names = ["Brightfield"]
    # Input scale with pixel sizes that differ from default config values
    mismatched_scale = [1, 1, 0.5, 0.2, 0.2]  # z=0.5, yx=0.2
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, 1, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=mismatched_scale)],
    )
    dataset.close()

    # Config with default pixel sizes (z=0.25, yx=0.1) — differ from input by 100%
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=channel_names,
        time_indices="all",
        reconstruction_dimension=3,
        phase=settings.PhaseSettings(),
    )
    config_path = tmp_path / "mismatch.yml"
    utils.model_to_yaml(recon_settings, config_path)

    runner = CliRunner()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        runner.invoke(
            cli,
            [
                "reconstruct",
                "-i",
                str(input_path / "0" / "0" / "0"),
                "-c",
                str(config_path),
                "-o",
                str(output_path),
            ],
            catch_exceptions=False,
        )

    mismatch_warnings = [w for w in caught if "do not match" in str(w.message).lower()]
    assert len(mismatch_warnings) > 0, "Expected pixel size mismatch warning"


def test_write_config_scale_to_output(tmp_path):
    """--write-config-scale-to-output should use config pixel sizes in the output."""
    input_path = tmp_path / "overwrite_input.zarr"
    output_path = tmp_path / "overwrite_output.zarr"

    channel_names = ["Brightfield"]
    original_scale = [1, 1, 0.5, 0.2, 0.2]
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, 1, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=original_scale)],
    )
    dataset.close()

    # Config with z=0.25, yx=0.1
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=channel_names,
        time_indices="all",
        reconstruction_dimension=3,
        phase=settings.PhaseSettings(),
    )
    config_path = tmp_path / "overwrite.yml"
    utils.model_to_yaml(recon_settings, config_path)

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--write-config-scale-to-output",
        ],
        catch_exceptions=False,
    )

    assert output_path.exists()
    with open_ome_zarr(output_path) as result:
        result_scale = result["0/0/0"].scale
        # T and C scales should be unchanged
        assert result_scale[0] == original_scale[0]
        assert result_scale[1] == original_scale[1]
        # Z, Y, X should match config pixel sizes
        assert result_scale[2] == 0.25  # z_pixel_size default
        assert result_scale[3] == 0.1  # yx_pixel_size default
        assert result_scale[4] == 0.1


def test_warn_pixel_size_mismatch_separate_y_and_x():
    """An anisotropic YXPixelSize raises separate y and x mismatches in the warning."""
    input_scale = (1.0, 1.0, 0.5, 0.4, 0.2)  # T, C, Z, Y, X
    config_pixel_sizes = (0.5, YXPixelSize(y=0.3, x=0.2))
    with pytest.warns(UserWarning) as record:
        _warn_pixel_size_mismatch(input_scale, config_pixel_sizes)
    text = "\n".join(str(w.message) for w in record)
    assert "y: input=0.4" in text and "config=0.3" in text
    assert "x:" not in text  # x matches, so should be omitted


def test_write_config_scale_to_output_anisotropic(tmp_path):
    """An anisotropic yx_pixel_size config writes distinct y and x scales to the output."""
    input_path = tmp_path / "aniso_input.zarr"
    output_path = tmp_path / "aniso_output.zarr"

    channel_names = ["GFP"]
    original_scale = [1, 1, 0.5, 0.4, 0.4]
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, 1, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=original_scale)],
    )
    dataset.close()

    fluor_tf = fluorescence_api.TransferFunctionSettings(
        yx_pixel_size={"y": 0.3, "x": 0.25},
        z_pixel_size=0.5,
        wavelength_emission=0.532,
    )
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=channel_names,
        time_indices="all",
        reconstruction_dimension=3,
        fluorescence=settings.FluorescenceSettings(transfer_function=fluor_tf),
    )
    config_path = tmp_path / "aniso.yml"
    utils.model_to_yaml(recon_settings, config_path)

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--write-config-scale-to-output",
        ],
        catch_exceptions=False,
    )

    with open_ome_zarr(output_path) as result:
        result_scale = result["0/0/0"].scale
        assert result_scale[2] == 0.5  # z
        assert result_scale[3] == 0.3  # y
        assert result_scale[4] == 0.25  # x


def test_warn_pixel_size_mismatch_isotropic_silent_when_equal():
    """No warning when input scale matches the config (isotropic case)."""
    input_scale = (1.0, 1.0, 0.5, 0.1, 0.1)
    config_pixel_sizes = (0.5, 0.1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_pixel_size_mismatch(input_scale, config_pixel_sizes)
    assert all("Input pixel sizes" not in str(w.message) for w in record)


def _phase_inputs(
    tmp_path,
    data=None,
    transform=None,
    version="0.5",
    name="input",
    recon_settings=None,
    tf_settings=None,
):
    """Write a one-position BF plate, a 3D phase config, and its transfer function.

    ``tf_settings`` computes the transfer function from a different config than
    the one returned, to fake a recomputed transfer function.
    """
    input_path = tmp_path / f"{name}.zarr"
    if data is None:
        data = np.random.default_rng(0).uniform(1, 100, size=(3, 1, 6, 8, 10)).astype(np.float32)
    if transform is None:
        transform = [TransformationMeta(type="scale", scale=[1, 1, 0.25, 0.1, 0.1])]
    with open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=["BF"], version=version) as dataset:
        dataset.create_position("0", "0", "0").create_image("0", data, transform=transform)

    def write_config(path, phase_settings):
        utils.model_to_yaml(
            settings.ReconstructionSettings(
                input_channel_names=["BF"], reconstruction_dimension=3, phase=phase_settings
            ),
            path,
        )
        return path

    config_path = write_config(tmp_path / f"{name}.yml", recon_settings or settings.PhaseSettings())
    tf_config_path = config_path if tf_settings is None else write_config(tmp_path / f"{name}_tf.yml", tf_settings)
    tf_path = tmp_path / f"{name}_tf.zarr"
    CliRunner().invoke(
        cli,
        ["compute-tf", "-i", str(input_path / "0" / "0" / "0"), "-c", str(tf_config_path), "-o", str(tf_path)],
        catch_exceptions=False,
    )
    return input_path / "0" / "0" / "0", config_path, tf_path


def _reconstruct(position_path, tf_path, config_path, result_path, num_processes=1, resume=False):
    apply_inverse_transfer_function_cli(
        [position_path], tf_path, config_path, result_path, num_processes, resume=resume
    )
    with open_ome_zarr(result_path / "0" / "0" / "0") as result:
        return result["0"][:]


def test_apply_inv_tf_process_pool_matches_serial(tmp_path):
    """Pool workers receive the transfer function as shared tensors; the result
    must match the serial path."""
    position_path, config_path, tf_path = _phase_inputs(tmp_path)

    serial = _reconstruct(position_path, tf_path, config_path, tmp_path / "serial.zarr", num_processes=1)
    pooled = _reconstruct(position_path, tf_path, config_path, tmp_path / "pooled.zarr", num_processes=2)

    assert np.any(serial != 0)
    np.testing.assert_array_equal(pooled, serial)


def test_apply_inverse_czyx_labels_the_volume_for_the_model():
    import xarray as xr

    from waveorder.cli.utils import apply_inverse_czyx

    czyx = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    coords = {
        "c": ("c", ["a", "b"]),
        "z": ("z", np.arange(3) * 0.5, {"units": "micrometer"}),
        "y": ("y", np.arange(4) * 0.1, {}),
        "x": ("x", np.arange(5) * 0.1, {}),
    }
    seen = {}

    def model_function(czyx_data, scale):
        seen["type"] = type(czyx_data)
        seen["c"] = list(czyx_data.coords["c"].values)
        seen["z_units"] = czyx_data.coords["z"].attrs["units"]
        return czyx_data * scale

    result = apply_inverse_czyx(czyx, model_function, coords, scale=2.0)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, czyx * 2)
    assert seen == {"type": xr.DataArray, "c": ["a", "b"], "z_units": "micrometer"}


def test_apply_inv_tf_writes_its_channels_by_name(tmp_path):
    """Output channels are looked up by name, so a config writing into a plate
    whose first channel belongs to another config leaves that channel alone."""
    from iohub.ngff.utils import create_empty_plate

    from waveorder.cli.apply_inverse_transfer_function import apply_inverse_transfer_function_single_position

    position_path, config_path, tf_path = _phase_inputs(tmp_path)
    reference = _reconstruct(position_path, tf_path, config_path, tmp_path / "reference.zarr")

    shared_path = tmp_path / "shared.zarr"
    create_empty_plate(
        store_path=shared_path,
        position_keys=[("0", "0", "0")],
        channel_names=["Other", "Phase3D"],
        shape=(3, 2, 6, 8, 10),
        scale=(1, 1, 0.25, 0.1, 0.1),
    )
    apply_inverse_transfer_function_single_position(
        position_path, tf_path, config_path, shared_path / "0" / "0" / "0", 1, ["Phase3D"]
    )

    with open_ome_zarr(shared_path / "0" / "0" / "0") as shared:
        np.testing.assert_array_equal(shared["0"][:, 0], 0)
        np.testing.assert_array_equal(shared["0"][:, 1], reference[:, 0])


def test_apply_inv_tf_addresses_timepoints_by_index(tmp_path):
    """A time_indices subset lands on the same indices in the output, even when
    the input's time axis carries a translation (which the old coordinate
    round-trip turned into a shifted index)."""
    data = np.random.default_rng(1).uniform(1, 100, size=(3, 1, 6, 8, 10)).astype(np.float32)
    reference_path, config_path, tf_path = _phase_inputs(tmp_path, data=data, name="reference")
    reference = _reconstruct(reference_path, tf_path, config_path, tmp_path / "reference_out.zarr")

    translated_path, translated_config, translated_tf = _phase_inputs(
        tmp_path,
        data=data,
        name="translated",
        transform=[
            TransformationMeta(type="scale", scale=[2.0, 1, 0.25, 0.1, 0.1]),
            TransformationMeta(type="translation", translation=[5.0, 0, 0, 0, 0]),
        ],
    )
    config = utils.yaml_to_model(translated_config, settings.ReconstructionSettings)
    config.time_indices = [1, 2]
    utils.model_to_yaml(config, translated_config)

    result = _reconstruct(translated_path, translated_tf, translated_config, tmp_path / "translated_out.zarr")

    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(result[1:], reference[1:])


def test_apply_inv_tf_skips_a_timepoint_with_a_blank_channel(tmp_path):
    """iohub's rule: a timepoint is skipped when any input channel is all zeros
    or NaNs, not only when all of them are."""
    input_path = tmp_path / "input.zarr"
    channel_names = [f"State{i}" for i in range(4)]
    data = np.random.default_rng(2).uniform(1, 100, size=(2, 4, 4, 8, 10)).astype(np.float32)
    data[1, 2] = 0  # t=1: one blank polarization state, signal in the others
    with open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names) as dataset:
        dataset.create_position("0", "0", "0").create_image("0", data)
    config_path = tmp_path / "biref.yml"
    utils.model_to_yaml(
        settings.ReconstructionSettings(
            input_channel_names=channel_names,
            reconstruction_dimension=3,
            birefringence=settings.BirefringenceSettings(),
        ),
        config_path,
    )
    tf_path = tmp_path / "tf.zarr"
    position_path = input_path / "0" / "0" / "0"
    CliRunner().invoke(
        cli,
        ["compute-tf", "-i", str(position_path), "-c", str(config_path), "-o", str(tf_path)],
        catch_exceptions=False,
    )

    result = _reconstruct(position_path, tf_path, config_path, tmp_path / "out.zarr")

    assert np.any(result[0] != 0)
    np.testing.assert_array_equal(result[1], 0)


def test_apply_inv_tf_resume(tmp_path):
    """resume skips finished timepoints, and recomputes after a settings change
    or a recomputed transfer function."""
    position_path, config_path, tf_path = _phase_inputs(tmp_path)
    result_path = tmp_path / "out.zarr"
    first = _reconstruct(position_path, tf_path, config_path, result_path)

    def mark_t0():
        with open_ome_zarr(result_path / "0" / "0" / "0", mode="r+") as result:
            result["0"][0] = 123.0

    # unchanged inputs: every timepoint is already finished
    mark_t0()
    resumed = _reconstruct(position_path, tf_path, config_path, result_path, resume=True)
    np.testing.assert_array_equal(resumed[0], 123.0)
    np.testing.assert_array_equal(resumed[1:], first[1:])

    # without resume, everything is recomputed
    recomputed = _reconstruct(position_path, tf_path, config_path, result_path, resume=False)
    np.testing.assert_array_equal(recomputed, first)

    # changed reconstruction settings: recomputed despite resume
    mark_t0()
    changed_settings = settings.PhaseSettings()
    changed_settings.apply_inverse.regularization_strength = 0.1
    _, changed_config, _ = _phase_inputs(tmp_path, name="changed", recon_settings=changed_settings)
    result = _reconstruct(position_path, tf_path, changed_config, result_path, resume=True)
    assert not np.any(result[0] == 123.0)

    # a transfer function recomputed with other settings: recomputed despite resume
    _reconstruct(position_path, tf_path, config_path, result_path)
    mark_t0()
    other_tf_settings = settings.PhaseSettings()
    other_tf_settings.transfer_function.wavelength_illumination = 0.6
    _, _, other_tf = _phase_inputs(tmp_path, name="other_tf", tf_settings=other_tf_settings)
    result = _reconstruct(position_path, other_tf, config_path, result_path, resume=True)
    assert not np.any(result[0] == 123.0)


def test_apply_inv_tf_resume_on_v04_output_recomputes_and_says_so(tmp_path, capsys):
    position_path, config_path, tf_path = _phase_inputs(tmp_path, version="0.4")
    result_path = tmp_path / "out.zarr"
    first = _reconstruct(position_path, tf_path, config_path, result_path)
    with open_ome_zarr(result_path / "0" / "0" / "0", mode="r+") as result:
        result["0"][0] = 123.0
    capsys.readouterr()

    resumed = _reconstruct(position_path, tf_path, config_path, result_path, resume=True)

    assert "OME-Zarr v0.5" in capsys.readouterr().out
    np.testing.assert_array_equal(resumed, first)
