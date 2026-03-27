"""CLI commands for the benchmarking suite.

Available as ``wo benchmark`` / ``wo bm``.
"""

import json
from pathlib import Path

import click

from benchmarks.config import load_experiment, resolve_recon_config
from benchmarks.runner import run_synthetic_case
from benchmarks.utils import collect_metadata, render_histogram

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
    """Run benchmark cases."""
    if experiment is None:
        experiment = str(_DEFAULT_EXPERIMENT)

    experiment_path = Path(experiment)
    output_dir = Path(output_dir)
    exp = load_experiment(experiment_path)

    metadata = collect_metadata()
    run_name = f"{metadata['git_hash']}_{exp.name}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    click.echo(click.style(f"WaveOrder Benchmark — {exp.name}", fg="green", bold=True))
    click.echo(f"  Git: {metadata['git_hash']} ({metadata['git_branch']}){' dirty' if metadata['git_dirty'] else ''}")
    click.echo(f"  Experiment: {exp.name} ({len(exp.cases)} cases)")
    click.echo()

    results = {}
    for case_name, case in exp.cases.items():
        if case.type != "synthetic" and scope == "synthetic":
            click.echo(f"  — {case_name:30s} skipped (non-synthetic)")
            continue
        if case.type == "hpc":
            click.echo(f"  — {case_name:30s} skipped (HPC not yet supported)")
            continue

        case_dir = run_dir / "cases" / case_name
        recon_config = resolve_recon_config(case, experiment_path.parent)
        modality = "phase" if "phase" in recon_config else "fluorescence"

        click.echo(f"  ▸ {case_name:30s} ", nl=False)
        try:
            metrics = run_synthetic_case(
                phantom_config=case.phantom,
                recon_config=recon_config,
                case_dir=case_dir,
                modality=modality,
            )
            results[case_name] = metrics

            # Read timing
            timing = json.loads((case_dir / "timing.json").read_text())
            total_s = timing.get("elapsed_s", 0)

            iq = metrics["image_quality"]
            line = f"done ({total_s:.1f}s)  midband={iq['midband_power']:.4g}"
            if "with_phantom" in metrics:
                line += f"  ssim={metrics['with_phantom']['ssim']:.3f}"
            click.echo(click.style(line, fg="green"))
        except Exception as e:
            click.echo(click.style(f"FAILED: {e}", fg="red"))
            results[case_name] = {"error": str(e)}

    click.echo()

    # Summary table
    _print_summary(results)

    # Save summary
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))
    click.echo(f"\n  Output: {run_dir}")


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Root output directory.")
def latest(output_dir):
    """Show summary of the most recent benchmark run."""
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

        # Histograms
        click.echo()
        for case_name, metrics in results.items():
            if "error" in metrics:
                continue
            hist = metrics.get("image_quality", {}).get("histogram")
            if hist:
                sparkline = render_histogram(hist)
                click.echo(f"  {case_name:30s} {sparkline}")


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


def _print_summary(results: dict):
    """Print a summary table of benchmark results."""
    header = f"  {'case':30s} {'midband':>10s} {'TV':>10s} {'ssim':>8s} {'mse':>10s}"
    click.echo(header)
    click.echo("  " + "─" * (len(header) - 2))

    for case_name, metrics in results.items():
        if "error" in metrics:
            click.echo(f"  {case_name:30s} {'ERROR':>10s}")
            continue

        iq = metrics.get("image_quality", {})
        mbp = f"{iq.get('midband_power', 0):.4g}"
        tv = f"{iq.get('total_variation', 0):.4g}"

        phantom = metrics.get("with_phantom", {})
        ssim_str = f"{phantom['ssim']:.3f}" if "ssim" in phantom else "—"
        mse_str = f"{phantom['mse']:.4g}" if "mse" in phantom else "—"

        click.echo(f"  {case_name:30s} {mbp:>10s} {tv:>10s} {ssim_str:>8s} {mse_str:>10s}")
