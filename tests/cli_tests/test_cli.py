import pytest
from typer.testing import CliRunner

from waveorder.cli.main import _COMMANDS, app


def test_main():
    runner = CliRunner()
    result = runner.invoke(app)

    assert result.exit_code == 2
    assert "waveorder" in result.output


def test_root_help():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "command",
    [
        *(name for name in _COMMANDS),
        *(alias for _, alias in _COMMANDS.values()),
        "benchmark",
        "bm",
    ],
)
def test_command_help(command):
    result = CliRunner().invoke(app, [command, "--help"])

    assert result.exit_code == 0, result.output
