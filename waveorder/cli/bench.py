"""CLI commands for the benchmarking suite.

Available as ``wo benchmark`` / ``wo bm``.
"""

import json
import os
import shutil
from pathlib import Path

import click

_DEFAULT_EXPERIMENT = Path(__file__).parent.parent.parent / "benchmarks" / "experiments" / "regression.yml"


def _resolve_output_dir(cli_value: str | None) -> Path:
    """Resolve the output directory.

    Precedence: CLI argument > ``WAVEORDER_BENCH_OUTPUT`` env var > ``"."``.
    """
    if cli_value is not None:
        return Path(cli_value)
    env_value = os.environ.get("WAVEORDER_BENCH_OUTPUT")
    if env_value:
        return Path(env_value)
    return Path(".")


def _output_dir_help() -> str:
    """Build --output-dir help text showing current env var value."""
    env_value = os.environ.get("WAVEORDER_BENCH_OUTPUT")
    if env_value:
        return f"Root output directory. [default: $WAVEORDER_BENCH_OUTPUT = {env_value}]"
    return "Root output directory. [default: $WAVEORDER_BENCH_OUTPUT if set, else '.']"


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
    default=None,
    help=_output_dir_help(),
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
    from benchmarks.runner import run_hpc_case, run_synthetic_case
    from benchmarks.utils import collect_metadata

    if experiment is None:
        experiment = str(_DEFAULT_EXPERIMENT)

    experiment_path = Path(experiment)
    output_dir = _resolve_output_dir(output_dir)
    exp = load_experiment(experiment_path)

    # Filter cases to those this scope will actually execute
    selected = {
        name: case
        for name, case in exp.cases.items()
        if not (case.type == "hpc" and scope == "synthetic") and case.type in ("synthetic", "hpc")
    }
    n_skipped_hpc = sum(1 for c in exp.cases.values() if c.type == "hpc" and scope == "synthetic")
    n_unsupported = sum(1 for c in exp.cases.values() if c.type not in ("synthetic", "hpc"))

    metadata = collect_metadata()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{timestamp}_{exp.name}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.copy2(experiment_path, run_dir / "experiment.yml")

    summary = f"{len(selected)} case{'s' if len(selected) != 1 else ''}"
    extras = []
    if n_skipped_hpc:
        extras.append(f"skipping {n_skipped_hpc} hpc")
    if n_unsupported:
        extras.append(f"skipping {n_unsupported} unsupported")
    if extras:
        summary += f"; {', '.join(extras)}"

    click.echo(click.style(f"WaveOrder Benchmark — {exp.name}", fg="green", bold=True))
    click.echo(f"  Git: {metadata['git_hash']} ({metadata['git_branch']}){' dirty' if metadata['git_dirty'] else ''}")
    click.echo(f"  Experiment: {exp.name} ({summary})")
    click.echo(f"  Output: {run_dir}")
    click.echo()
    _print_header()

    results = {}
    for case_name, case in selected.items():
        case_dir = run_dir / "cases" / case_name
        recon_config = resolve_recon_config(case, experiment_path.parent)

        click.echo(f"  {case_name:24s} running...", nl=False)
        try:
            if case.type == "synthetic":
                modality = "phase" if "phase" in recon_config else "fluorescence"
                metrics = run_synthetic_case(
                    phantom_config=case.phantom,
                    recon_config=recon_config,
                    case_dir=case_dir,
                    modality=modality,
                )
            elif case.type == "hpc":
                metrics = run_hpc_case(
                    input_path=case.input,
                    position=case.position,
                    recon_config=recon_config,
                    case_dir=case_dir,
                )

            results[case_name] = metrics

            timing = json.loads((case_dir / "timing.json").read_text())
            elapsed = timing.get("elapsed_s", 0)
            metrics["elapsed_s"] = elapsed
            click.echo("\033[2K\r", nl=False)
            _print_row(case_name, metrics, elapsed=elapsed)
        except Exception as e:
            click.echo("\033[2K\r", nl=False)
            click.echo(f"  {case_name:24s} " + click.style(f"FAILED: {e}", fg="red"))
            results[case_name] = {"error": str(e)}

    # Save summary
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None, help=_output_dir_help())
def latest(output_dir):
    """Show summary of the most recent benchmark run."""
    click.echo(click.style("Loading latest benchmark...", fg="green"))

    output_dir = _resolve_output_dir(output_dir)
    run_dirs = _list_run_dirs(output_dir)
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
@click.option("--output-dir", "-o", type=click.Path(), default=None, help=_output_dir_help())
@click.option("--limit", "-n", type=int, default=10, help="Number of runs to show.")
def history(output_dir, limit):
    """List recent benchmark runs."""
    output_dir = _resolve_output_dir(output_dir)
    run_dirs = _list_run_dirs(output_dir)
    if not run_dirs:
        click.echo("No benchmark runs found.")
        return

    for run_dir in run_dirs[-limit:]:
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summary_path = run_dir / "summary.json"
        n_cases = 0
        if summary_path.exists():
            n_cases = len(json.loads(summary_path.read_text()))
        click.echo(f"  {run_dir.name:50s} {n_cases} cases  git={meta.get('git_hash', '?')}")


_NON_RUN_DIRS = {"annotated_references"}


def _list_run_dirs(output_dir: str | Path) -> list[Path]:
    """Return all run directories under output_dir (oldest first)."""
    root = Path(output_dir)
    if not root.exists():
        return []
    candidates = [
        p for p in root.iterdir() if p.is_dir() and p.name not in _NON_RUN_DIRS and not p.name.startswith(".")
    ]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _find_runs(output_dir: str | Path, n: int = 1) -> list[Path]:
    """Return the N most recent run directories, newest first."""
    return list(reversed(_list_run_dirs(output_dir)))[:n]


def _load_summary(run_dir: Path) -> dict:
    """Load summary.json from a run directory."""
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@benchmark.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None, help=_output_dir_help())
@click.argument("run_a", required=False)
@click.argument("run_b", required=False)
def compare(output_dir, run_a, run_b):
    """Compare two benchmark runs side by side.

    If no run names given, compares the two most recent runs.

    \b
    Example:
      \033[92mwo bm compare\033[0m
    """
    output_dir = _resolve_output_dir(output_dir)

    if run_a and run_b:
        dirs = [output_dir / run_a, output_dir / run_b]
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
@click.option("--output-dir", "-o", type=click.Path(), default=None, help=_output_dir_help())
@click.argument("path", required=False, default=None)
def view(output_dir, path):
    """Open benchmark results in napari.

    PATH selects what to view: ``latest`` or a run name, optionally
    followed by ``/case_name``. Nothing opens the most recent run.
    Raw input and reconstruction are loaded as separate layers and
    tiled side-by-side via napari's grid mode.

    \b
    Example:
      \033[92mwo bm view\033[0m
      \033[92mwo bm view latest/phase_3d_beads\033[0m
      \033[92mwo bm view 2026-03-27T11-00-10_regression/reg_1e-3\033[0m
    """
    click.echo(click.style("Opening benchmark results...", fg="green"))

    # Deferred imports
    import napari
    from iohub.ngff import open_ome_zarr

    from benchmarks.config import load_experiment

    # Parse path: nothing, "run_name", or "run_name/case_name"
    run_name = None
    case = None
    if path:
        parts = path.split("/", 1)
        run_name = parts[0]
        if len(parts) > 1:
            case = parts[1]

    output_dir = _resolve_output_dir(output_dir)
    if run_name in (None, "latest"):
        runs = _find_runs(output_dir, n=1)
        if not runs:
            click.echo("No benchmark runs found.")
            return
        run_dir = runs[0]
    else:
        run_dir = output_dir / run_name
        if not run_dir.exists():
            click.echo(f"Run not found: {run_dir}")
            return

    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        click.echo(f"No cases in run {run_dir.name}.")
        return

    # Load the experiment YAML saved with this run to locate raw inputs
    exp_path = run_dir / "experiment.yml"
    experiment = load_experiment(exp_path) if exp_path.exists() else None

    viewer = napari.Viewer(title=f"wo bm view — {run_dir.name}")

    case_dirs = [cases_dir / case] if case else sorted(cases_dir.iterdir())
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name

        # Raw input: simulated.zarr for synthetic, {input}/{position} for hpc
        raw_path = _resolve_raw_path(case_dir, case_name, experiment)
        channel_filter = _input_channel_names(case_dir)
        if raw_path and raw_path.exists():
            _add_zarr_to_viewer(
                viewer,
                raw_path,
                f"{case_name}/raw",
                open_ome_zarr,
                channels=channel_filter,
            )
        else:
            click.echo(f"  {case_name}: no raw input found")

        # Reconstruction
        recon_path = case_dir / "reconstruction.zarr"
        if recon_path.exists():
            _add_zarr_to_viewer(viewer, recon_path, f"{case_name}/recon", open_ome_zarr)

    viewer.grid.enabled = True
    napari.run()


def _resolve_raw_path(case_dir: Path, case_name: str, experiment) -> Path | None:
    """Locate the raw input for a case.

    Synthetic cases store their simulated measurement as ``simulated.zarr``
    in the case directory. HPC cases point at an external position, which
    is reconstructed from the experiment YAML as ``{input}/{position}``.
    """
    sim_path = case_dir / "simulated.zarr"
    if sim_path.exists():
        return sim_path
    if experiment is None or case_name not in experiment.cases:
        return None
    case_cfg = experiment.cases[case_name]
    if case_cfg.type == "hpc" and case_cfg.input and case_cfg.position:
        return Path(case_cfg.input) / case_cfg.position
    return None


def _add_zarr_to_viewer(viewer, path, prefix, open_ome_zarr, channels=None):
    """Add channels from an OME-Zarr to the viewer.

    Handles both HCS (multi-position) and single-position (FOV) stores.
    If ``channels`` is given, only channels whose names match are added.
    """
    import numpy as np

    dataset = open_ome_zarr(path, mode="r")
    try:
        positions = [pos for _, pos in dataset.positions()]
    except (AttributeError, TypeError):
        positions = [dataset]

    for position in positions:
        data = np.array(position["0"][0])  # CZYX at T=0
        for c_idx, ch_name in enumerate(position.channel_names):
            if channels is not None and ch_name not in channels:
                continue
            ch_data = data[c_idx]
            if ch_data.shape[0] == 1:
                ch_data = ch_data[0]
            viewer.add_image(ch_data, name=f"{prefix}/{ch_name}")
    dataset.close()


def _input_channel_names(case_dir: Path) -> list[str] | None:
    """Read ``input_channel_names`` from the case's saved config.yml."""
    import yaml

    cfg_path = case_dir / "config.yml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    names = cfg.get("input_channel_names") if isinstance(cfg, dict) else None
    return list(names) if names else None


@benchmark.command()
@click.argument("case_name")
@click.option("--output-dir", "-o", type=click.Path(), default=None, help=_output_dir_help())
def mark(case_name, output_dir):
    """Mark the latest run's reconstruction as the annotated reference.

    Copies the reconstruction zarr to annotated_references/.

    \b
    Example:
      \033[92mwo bm mark phase_3d_beads\033[0m
    """
    output_dir = _resolve_output_dir(output_dir)
    runs = _find_runs(output_dir, n=1)
    if not runs:
        click.echo("No benchmark runs found.")
        return

    run_dir = runs[0]
    recon_path = run_dir / "cases" / case_name / "reconstruction.zarr"
    if not recon_path.exists():
        click.echo(f"No reconstruction found for case '{case_name}' in {run_dir.name}")
        return

    ref_dir = output_dir / "annotated_references"
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
