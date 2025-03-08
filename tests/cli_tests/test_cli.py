from click.testing import CliRunner

from waveorder.cli.main import cli


def test_main():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0
    assert "Toolkit" in result.output
