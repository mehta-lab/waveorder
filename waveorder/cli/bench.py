"""CLI commands for the benchmarking suite.

Available as ``wo benchmark`` / ``wo bm``.
"""

import json
from pathlib import Path

import click

_DEFAULT_EXPERIMENT = Path(__file__).parent.parent.parent / "benchmarks" / "experiments" / "regression.yml"


@click.group("benchmark")
def benchmark():
    """Run and inspect reconstruction benchmarks."""
    pass


@benchmark.command()
@click.option(
    "--experiment",
    "-e",
    type=click.Path(exists=True),
    default=None,
    help="Path to experiment YAML. Default: regression suite.",
)
@click.option(
    "--scope",
    type=click.Choice(["synthetic", "all"]),
    default="synthetic",
    help="Which cases to run.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Root output directory for benchmark runs.",
)
def run(experiment, scope, output_dir):
    """Run benchmark cases.

    \b
    Example:
      \033[92mwo bm run --scope synthetic\033[0m
    """
    click.echo(click.style("Starting benchmark run...", fg="green"))

    # Deferred imports: benchmarks.runner pulls in torch, iohub, etc.
    from benchmarks.config import load_experiment, resolve_recon_config
    from benchmarks.runner import run_synthetic_case
    from benchmarks.utils import collect_metadata

    if experiment is None:
        experiment = str(_DEFAULT_EXPERIMENT)

    experiment_path = Path(experiment)
    output_dir = Path(output_dir)
    exp = load_experiment(experiment_path)

    metadata = collect_metadata()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{timestamp}_{exp.name}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    click.echo(click.style(f"WaveOrder Benchmark — {exp.name}", fg="green", bold=True))
    click.echo(f"  Git: {metadata['git_hash']} ({metadata['git_branch']}){' dirty' if metadata['git_dirty'] else ''}")
    click.echo(f"  Experiment: {exp.name} ({len(exp.cases)} cases)")
    click.echo(f"  Output: {run_dir}")
    click.echo()
    _print_header()

    results = {}
    for case_name, case in exp.cases.items():
        if case.type != "synthetic" and scope == "synthetic":
            click.echo(f"  {case_name:24s} skipped (non-synthetic)")
            continue
        if case.type == "hpc":
            click.echo(f"  {case_name:24s} skipped (HPC not yet supported)")
            continue

        case_dir = run_dir / "cases" / case_name
        recon_config = resolve_recon_config(case, experiment_path.parent)
        modality = "phase" if "phase" in recon_config else "fluorescence"

        click.echo(f"  {case_name:24s} running...", nl=False)
        try:
            metrics = run_synthetic_case(
                phantom_config=case.phantom,
                recon_config=recon_config,
                case_dir=case_dir,
                modality=modality,
            )
            results[case_name] = metrics

            timing = json.loads((case_dir / "timing.json").read_text())
            elapsed = timing.get("elapsed_s", 0)
            metrics["elapsed_s"] = elapsed
            # Clear the "running..." line and print the result row
            click.echo("\033[2K\r", nl=False)
            _print_row(case_name, metrics, elapsed=elapsed)
        except Exception as e:
            click.echo("\033[2K\r", nl=False)
            click.echo(f"  {case_name:24s} " + click.style(f"FAILED: {e}", fg="red"))
            results[case_name] = {"error": str(e)}

    # Save summary
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Root output directory.")
def latest(output_dir):
    """Show summary of the most recent benchmark run."""
    click.echo(click.style("Loading latest benchmark...", fg="green"))

    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        click.echo("No benchmark runs found.")
        return

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        click.echo("No benchmark runs found.")
        return

    run_dir = run_dirs[-1]
    click.echo(click.style(f"Latest run: {run_dir.name}", fg="green", bold=True))

    # Metadata
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        click.echo(f"  Git: {meta.get('git_hash', '?')} ({meta.get('git_branch', '?')})")

    # Summary
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        results = json.loads(summary_path.read_text())
        click.echo()
        _print_summary(results)


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Root output directory.")
@click.option("--limit", "-n", type=int, default=10, help="Number of runs to show.")
def history(output_dir, limit):
    """List recent benchmark runs."""
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        click.echo("No benchmark runs found.")
        return

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs[:limit]:
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summary_path = run_dir / "summary.json"
        n_cases = 0
        if summary_path.exists():
            n_cases = len(json.loads(summary_path.read_text()))
        click.echo(f"  {run_dir.name:50s} {n_cases} cases  git={meta.get('git_hash', '?')}")


def _fmt(value, width=10):
    """Format a float as fixed-width scientific notation with explicit sign."""
    if value == 0:
        return f"{' 0':>{width}s}"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1e}".rjust(width)


def _sparkline(hist, n_bins=10):
    """Render a small inline sparkline (bars only)."""
    import numpy as np

    counts = np.array(hist["counts"], dtype=float)
    if len(counts) > n_bins:
        factor = len(counts) // n_bins
        counts = counts[: factor * n_bins].reshape(n_bins, factor).sum(axis=1)
    if counts.max() == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    normalized = counts / counts.max()
    return "".join(blocks[int(v * 7)] for v in normalized)


_W = 10  # metric column width
_H = 10  # min/max column width


def _print_header():
    """Print the summary table header."""
    header = (
        f"  {'case':24s} {'time':>6s}"
        f" {'midband':>{_W}s} {'mse':>{_W}s} {'ssim':>{_W}s}"
        f"  {'min':>{_H}s} {'histogram':^12s} {'max':>{_H}s}"
    )
    click.echo(header)
    click.echo("  " + "─" * (len(header) - 2))


def _print_row(case_name: str, metrics: dict, elapsed: float | None = None):
    """Print a single summary table row."""
    iq = metrics.get("image_quality", {})
    mbp = _fmt(iq.get("midband_power", 0), _W)

    phantom = metrics.get("with_phantom", {})
    ssim_str = f"{phantom['ssim']:>{_W}.3f}" if "ssim" in phantom else f"{'—':>{_W}s}"
    mse_str = _fmt(phantom["mse"], _W) if "mse" in phantom else f"{'—':>{_W}s}"

    hist = iq.get("histogram", {})
    if hist:
        import numpy as np

        edges = np.array(hist["bin_edges"])
        min_str = _fmt(edges[0], _H)
        max_str = _fmt(edges[-1], _H)
        spark = _sparkline(hist)
    else:
        min_str = f"{'—':>{_H}s}"
        max_str = f"{'—':>{_H}s}"
        spark = ""

    time_str = f"{elapsed:5.1f}s" if elapsed is not None else f"{'—':>6s}"

    click.echo(f"  {case_name:24s} {time_str} {mbp} {mse_str} {ssim_str}  {min_str} {spark:^12s} {max_str}")


def _print_summary(results: dict):
    """Print a full summary table (header + all rows)."""
    _print_header()
    for case_name, metrics in results.items():
        if "error" in metrics:
            click.echo(f"  {case_name:24s} " + click.style("ERROR", fg="red"))
            continue
        _print_row(case_name, metrics, elapsed=metrics.get("elapsed_s"))
