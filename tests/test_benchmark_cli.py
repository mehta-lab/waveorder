"""Tests for the ``wo bm`` click commands."""

import json
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from waveorder.cli.bench import benchmark


def _make_run_dir(base, name, metadata=None, summary=None):
    d = base / name
    d.mkdir(parents=True)
    if metadata is not None:
        (d / "metadata.json").write_text(json.dumps(metadata))
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    return d


class TestHistory:
    def test_empty(self, tmp_path):
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output

    def test_lists_runs(self, tmp_path):
        _make_run_dir(
            tmp_path,
            "2026-04-20T10-00-00_regression",
            metadata={"git_hash": "abc123"},
            summary={"case_a": {}},
        )
        _make_run_dir(
            tmp_path,
            "2026-04-20T11-00-00_regression",
            metadata={"git_hash": "def456"},
            summary={"case_a": {}, "case_b": {}},
        )
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "2026-04-20T10-00-00_regression" in result.output
        assert "2026-04-20T11-00-00_regression" in result.output


class TestLatest:
    def test_empty(self, tmp_path):
        result = CliRunner().invoke(benchmark, ["latest", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output

    def test_shows_latest(self, tmp_path):
        summary = {
            "case_a": {
                "image_quality": {
                    "midband_power": 0.01,
                    "histogram": {"bin_edges": [0.0, 0.5, 1.0], "counts": [5, 10]},
                },
                "elapsed_s": 2.5,
            },
        }
        _make_run_dir(
            tmp_path,
            "2026-04-20T10-00-00_regression",
            metadata={"git_hash": "abc", "git_branch": "main"},
            summary=summary,
        )
        result = CliRunner().invoke(benchmark, ["latest", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "2026-04-20T10-00-00_regression" in result.output
        assert "case_a" in result.output


class TestCompare:
    def test_needs_two_runs(self, tmp_path):
        result = CliRunner().invoke(benchmark, ["compare", "-o", str(tmp_path)])
        assert "Need at least 2 runs" in result.output

    def test_compares_two_latest(self, tmp_path):
        summary_a = {"case_a": {"image_quality": {"midband_power": 0.01}, "with_phantom": {"mse": 0.1, "ssim": 0.9}}}
        summary_b = {"case_a": {"image_quality": {"midband_power": 0.02}, "with_phantom": {"mse": 0.05, "ssim": 0.95}}}
        _make_run_dir(tmp_path, "2026-04-20T10-00-00_regression", metadata={"git_hash": "a"}, summary=summary_a)
        _make_run_dir(tmp_path, "2026-04-20T11-00-00_regression", metadata={"git_hash": "b"}, summary=summary_b)
        result = CliRunner().invoke(benchmark, ["compare", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "case_a" in result.output
        assert "ssim" in result.output


class TestOutputDirResolution:
    def test_cli_arg_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WAVEORDER_BENCH_OUTPUT", "/nonexistent/path")
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output

    def test_env_var_used_when_no_cli(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WAVEORDER_BENCH_OUTPUT", str(tmp_path))
        result = CliRunner().invoke(benchmark, ["history"])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output


class TestRun:
    """Mock-driven tests of the run command's orchestration."""

    def _write_experiment(self, tmp_path, cases):
        exp = {"name": "test_exp", "cases": cases}
        exp_yml = tmp_path / "exp.yml"
        exp_yml.write_text(yaml.dump(exp))
        (tmp_path / "recon.yml").write_text(yaml.dump({"phase": {"transfer_function": {}}}))
        return exp_yml

    def test_synthetic_scope_skips_hpc(self, tmp_path):
        exp_yml = self._write_experiment(
            tmp_path,
            {
                "syn_case": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
                    "config": "recon.yml",
                },
                "hpc_case": {
                    "type": "hpc",
                    "input": "/fake/input.zarr",
                    "position": "A/1/000000",
                    "config": "recon.yml",
                },
            },
        )

        with (
            patch("benchmarks.runner.run_synthetic_case") as mock_syn,
            patch("benchmarks.runner.run_hpc_case") as mock_hpc,
        ):
            mock_syn.return_value = {"image_quality": {"midband_power": 0.01}}
            mock_hpc.return_value = {"image_quality": {"midband_power": 0.01}}
            with CliRunner().isolated_filesystem():
                # Need to create timing.json for the bench.run post-case readback
                def _side_effect(**kwargs):
                    case_dir = kwargs["case_dir"]
                    case_dir.mkdir(parents=True, exist_ok=True)
                    (case_dir / "timing.json").write_text(json.dumps({"elapsed_s": 0.1}))
                    return {"image_quality": {"midband_power": 0.01}}

                mock_syn.side_effect = _side_effect
                result = CliRunner().invoke(
                    benchmark,
                    ["run", "-e", str(exp_yml), "--scope", "synthetic", "-o", str(tmp_path)],
                )

        assert result.exit_code == 0, result.output
        assert mock_syn.call_count == 1
        assert mock_hpc.call_count == 0
        assert "skipping 1 hpc" in result.output

    def test_all_scope_runs_hpc(self, tmp_path):
        exp_yml = self._write_experiment(
            tmp_path,
            {
                "hpc_case": {
                    "type": "hpc",
                    "input": "/fake/input.zarr",
                    "position": "A/1/000000",
                    "config": "recon.yml",
                },
            },
        )

        def _side_effect(**kwargs):
            case_dir = kwargs["case_dir"]
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "timing.json").write_text(json.dumps({"elapsed_s": 0.1}))
            return {"image_quality": {"midband_power": 0.01}}

        with patch("benchmarks.runner.run_hpc_case") as mock_hpc:
            mock_hpc.side_effect = _side_effect
            result = CliRunner().invoke(
                benchmark,
                ["run", "-e", str(exp_yml), "--scope", "all", "-o", str(tmp_path)],
            )
        assert result.exit_code == 0, result.output
        assert mock_hpc.call_count == 1

    def test_records_error_on_failure(self, tmp_path):
        exp_yml = self._write_experiment(
            tmp_path,
            {
                "bad_case": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
                    "config": "recon.yml",
                },
            },
        )

        with patch("benchmarks.runner.run_synthetic_case") as mock_syn:
            mock_syn.side_effect = RuntimeError("boom")
            result = CliRunner().invoke(
                benchmark,
                ["run", "-e", str(exp_yml), "--scope", "synthetic", "-o", str(tmp_path)],
            )

        assert result.exit_code == 0  # run continues past case failures
        assert "FAILED" in result.output
        # summary.json records the error
        runs = [p for p in tmp_path.iterdir() if p.is_dir() and "test_exp" in p.name]
        assert len(runs) == 1
        summary = json.loads((runs[0] / "summary.json").read_text())
        assert "error" in summary["bad_case"]
