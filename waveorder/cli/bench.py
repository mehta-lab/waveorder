"""CLI commands for the benchmarking suite.

Available as ``wo benchmark`` / ``wo bm``.
"""

import json
import os
import shutil
from pathlib import Path

import click

_DEFAULT_EXPERIMENT = Path(__file__).parent.parent.parent / "benchmarks" / "experiments" / "regression.yml"
_DEFAULT_OUTPUT_DIR = os.environ.get("WAVEORDER_BENCH_OUTPUT", ".")


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
    shutil.copy2(experiment_path, run_dir / "experiment.yml")

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
@click.option("--output-dir", "-o", type=click.Path(), default=_DEFAULT_OUTPUT_DIR, help="Root output directory.")
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
@click.option("--output-dir", "-o", type=click.Path(), default=_DEFAULT_OUTPUT_DIR, help="Root output directory.")
@click.option("--limit", "-n", type=int, default=10, help="Number of runs to show.")
def history(output_dir, limit):
    """List recent benchmark runs."""
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        click.echo("No benchmark runs found.")
        return

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    for run_dir in run_dirs[-limit:]:
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summary_path = run_dir / "summary.json"
        n_cases = 0
        if summary_path.exists():
            n_cases = len(json.loads(summary_path.read_text()))
        click.echo(f"  {run_dir.name:50s} {n_cases} cases  git={meta.get('git_hash', '?')}")


def _find_runs(output_dir: str | Path, n: int = 1) -> list[Path]:
    """Return the N most recent run directories, newest first."""
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:n]


def _load_summary(run_dir: Path) -> dict:
    """Load summary.json from a run directory."""
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=_DEFAULT_OUTPUT_DIR, help="Root output directory.")
@click.argument("run_a", required=False)
@click.argument("run_b", required=False)
def compare(output_dir, run_a, run_b):
    """Compare two benchmark runs side by side.

    If no run names given, compares the two most recent runs.

    \b
    Example:
      \033[92mwo bm compare\033[0m
    """
    runs_dir = Path(output_dir) / "runs"

    if run_a and run_b:
        dirs = [runs_dir / run_a, runs_dir / run_b]
    else:
        dirs = _find_runs(output_dir, n=2)

    if len(dirs) < 2:
        click.echo("Need at least 2 runs to compare.")
        return

    dir_a, dir_b = dirs[1], dirs[0]  # older first
    summary_a = _load_summary(dir_a)
    summary_b = _load_summary(dir_b)

    click.echo(click.style("Benchmark comparison", fg="green", bold=True))
    click.echo(f"  A: {dir_a.name}")
    click.echo(f"  B: {dir_b.name}")
    click.echo()

    # Find common cases
    common = set(summary_a.keys()) & set(summary_b.keys())
    if not common:
        click.echo("No common cases between runs.")
        return

    # Header
    click.echo(f"  {'case':24s} {'metric':>10s} {'A':>10s} {'B':>10s} {'delta':>10s}")
    click.echo("  " + "─" * 66)

    for case_name in sorted(common):
        ma, mb = summary_a[case_name], summary_b[case_name]
        if "error" in ma or "error" in mb:
            continue

        # Compare key metrics
        pairs = []
        iq_a, iq_b = ma.get("image_quality", {}), mb.get("image_quality", {})
        for key in ["midband_power"]:
            if key in iq_a and key in iq_b:
                pairs.append(("midband", iq_a[key], iq_b[key]))

        for group in ["with_phantom", "with_reference"]:
            ga, gb = ma.get(group, {}), mb.get(group, {})
            for key in ["mse", "ssim"]:
                if key in ga and key in gb:
                    pairs.append((key, ga[key], gb[key]))

        for metric, va, vb in pairs:
            delta = vb - va
            click.echo(f"  {case_name:24s} {metric:>10s} {_fmt(va)} {_fmt(vb)} {_fmt(delta)}")
            case_name = ""  # only show name on first row


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=_DEFAULT_OUTPUT_DIR, help="Root output directory.")
@click.argument("path", required=False, default=None)
def view(output_dir, path):
    """Open benchmark results in napari.

    PATH selects what to view: a run name, a case name (run/case),
    or nothing for the latest run.

    \b
    Example:
      \033[92mwo bm view\033[0m
      \033[92mwo bm view 2026-03-27T11-00-10_regression\033[0m
      \033[92mwo bm view 2026-03-27T11-00-10_regression/reg_1e-3\033[0m
    """
    click.echo(click.style("Opening benchmark results...", fg="green"))

    # Deferred imports
    import napari
    from iohub.ngff import open_ome_zarr

    # Parse path: nothing, "run_name", or "run_name/case_name"
    run_name = None
    case = None
    if path:
        parts = path.split("/", 1)
        run_name = parts[0]
        if len(parts) > 1:
            case = parts[1]

    if run_name:
        run_dir = Path(output_dir) / "runs" / run_name
        if not run_dir.exists():
            click.echo(f"Run not found: {run_dir}")
            return
    else:
        runs = _find_runs(output_dir, n=1)
        if not runs:
            click.echo("No benchmark runs found.")
            return
        run_dir = runs[0]
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        click.echo("No cases in latest run.")
        return

    viewer = napari.Viewer(title=f"wo bm view — {run_dir.name}")

    case_dirs = [cases_dir / case] if case else sorted(cases_dir.iterdir())
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name

        # Load reconstruction
        recon_path = case_dir / "reconstruction.zarr"
        if recon_path.exists():
            _add_zarr_to_viewer(viewer, recon_path, f"{case_name}/recon", open_ome_zarr)

        # Load simulated data (synthetic cases)
        sim_path = case_dir / "simulated.zarr"
        if sim_path.exists():
            _add_zarr_to_viewer(viewer, sim_path, f"{case_name}/simulated", open_ome_zarr)

    napari.run()


def _add_zarr_to_viewer(viewer, path, prefix, open_ome_zarr):
    """Add all channels from an OME-Zarr to the viewer."""
    import numpy as np

    dataset = open_ome_zarr(path, mode="r")
    for _, position in dataset.positions():
        data = np.array(position["0"][0])  # CZYX
        for c_idx, ch_name in enumerate(position.channel_names):
            ch_data = data[c_idx]
            # Squeeze singleton Z
            if ch_data.shape[0] == 1:
                ch_data = ch_data[0]
            viewer.add_image(ch_data, name=f"{prefix}/{ch_name}")
    dataset.close()


@benchmark.command()
@click.argument("case_name")
@click.option("--output-dir", "-o", type=click.Path(), default=_DEFAULT_OUTPUT_DIR, help="Root output directory.")
def mark(case_name, output_dir):
    """Mark the latest run's reconstruction as the annotated reference.

    Copies the reconstruction zarr to annotated_references/.

    \b
    Example:
      \033[92mwo bm mark phase_3d_beads\033[0m
    """
    runs = _find_runs(output_dir, n=1)
    if not runs:
        click.echo("No benchmark runs found.")
        return

    run_dir = runs[0]
    recon_path = run_dir / "cases" / case_name / "reconstruction.zarr"
    if not recon_path.exists():
        click.echo(f"No reconstruction found for case '{case_name}' in {run_dir.name}")
        return

    ref_dir = Path(output_dir) / "annotated_references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    dest = ref_dir / f"{case_name}.zarr"

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(recon_path, dest)
    click.echo(click.style(f"Marked {case_name} reference: {dest}", fg="green"))


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
