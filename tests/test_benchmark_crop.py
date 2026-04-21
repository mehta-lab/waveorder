"""Tests for the case-level ``crop`` bbox feature."""

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from benchmarks.config import CaseConfig, CropConfig
from benchmarks.runner import crop_input_position


def _write_source_zarr(path, shape=(2, 7, 2048, 2048)):
    ds = open_ome_zarr(path, layout="hcs", mode="w", channel_names=["GFP", "BF"])
    pos = ds.create_position("A", "1", "029029")
    pos.create_zeros(
        "0",
        (1, *shape),
        dtype=np.float32,
        transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 2.0, 0.65, 0.65])],
    )
    pos["0"][0] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    ds.close()


class TestCropConfigSchema:
    def test_accepts_partial_axes(self):
        c = CropConfig(y=(100, 200))
        assert c.y == (100, 200)
        assert c.x is None
        assert c.z is None

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            CropConfig(y=(0, 10), bogus=1)


class TestCaseConfigWithCrop:
    def test_parses(self):
        case = CaseConfig(
            type="hpc",
            input="/a.zarr",
            position="A/1/000000",
            crop={"y": [10, 50], "x": [20, 100]},
        )
        assert case.crop.y == (10, 50)
        assert case.crop.x == (20, 100)
        assert case.crop.z is None

    def test_defaults_to_none(self):
        case = CaseConfig(type="synthetic")
        assert case.crop is None


class TestCropInputPosition:
    def test_writes_cropped_zarr(self, tmp_path):
        src = tmp_path / "source.zarr"
        _write_source_zarr(src, shape=(2, 7, 2048, 2048))

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        crop = CropConfig(y=(768, 1280), x=(768, 1280))
        out = crop_input_position(str(src), "A/1/029029", crop, case_dir)

        assert out == case_dir / "cropped_input.zarr"
        assert out.exists()

        # Inspect the cropped zarr
        dst = open_ome_zarr(out, layout="hcs", mode="r")
        dst_pos = list(dst.positions())[0][1]
        dst_data = np.array(dst_pos["0"][0])
        assert dst_data.shape == (2, 7, 512, 512)
        dst.close()

    def test_crop_on_z_axis(self, tmp_path):
        src = tmp_path / "source.zarr"
        _write_source_zarr(src, shape=(2, 7, 128, 128))

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        crop = CropConfig(z=(1, 4))
        out = crop_input_position(str(src), "A/1/029029", crop, case_dir)

        dst = open_ome_zarr(out, layout="hcs", mode="r")
        dst_pos = list(dst.positions())[0][1]
        assert np.array(dst_pos["0"][0]).shape == (2, 3, 128, 128)
        dst.close()

    def test_overwrites_existing(self, tmp_path):
        src = tmp_path / "source.zarr"
        _write_source_zarr(src, shape=(2, 7, 128, 128))
        case_dir = tmp_path / "case"
        case_dir.mkdir()

        crop1 = CropConfig(y=(0, 32))
        crop2 = CropConfig(y=(0, 64))
        crop_input_position(str(src), "A/1/029029", crop1, case_dir)
        out = crop_input_position(str(src), "A/1/029029", crop2, case_dir)

        dst = open_ome_zarr(out, layout="hcs", mode="r")
        dst_pos = list(dst.positions())[0][1]
        assert np.array(dst_pos["0"][0]).shape == (2, 7, 64, 128)
        dst.close()
