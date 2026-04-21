"""CLI commands for the benchmarking suite.

Available as ``wo benchmark`` / ``wo bm``.
"""

import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import yaml

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


def _output_dir_option():
    """The shared ``--output-dir`` click option used by every bm command."""
    return click.option(
        "--output-dir",
        "-o",
        type=click.Path(),
        default=None,
        help=_output_dir_help(),
    )


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
@_output_dir_option()
@click.option(
    "--save-all",
    is_flag=True,
    default=False,
    help=(
        "Keep every intermediate output. By default transfer function "
        "zarrs are always deleted and reconstruction/simulated zarrs "
        "larger than 25 MB are deleted after metrics are computed."
    ),
)
def run(experiment, scope, output_dir, save_all):
    """Run benchmark cases.

    By default only metrics, timing, configs, and small (<25 MB) output
    zarrs are kept — transfer function zarrs and oversized
    reconstruction/simulation zarrs are deleted after metrics are
    computed to keep the benchmarks folder small. Pass --save-all to
    keep every intermediate.

    \b
    Example:
      \033[92mwo bm run --scope synthetic\033[0m
      \033[92mwo bm run --save-all\033[0m
    """
    click.echo(click.style("Starting benchmark run...", fg="green"))

    # Deferred imports: benchmarks.runner pulls in torch, iohub, etc.
    from benchmarks.config import infer_modality, load_experiment, resolve_recon_config
    from benchmarks.runner import run_hpc_case, run_synthetic_case
    from benchmarks.utils import collect_metadata

    if experiment is None:
        experiment = str(_DEFAULT_EXPERIMENT)

    experiment_path = Path(experiment)
    output_dir = _resolve_output_dir(output_dir)
    exp = load_experiment(experiment_path)

    selected = {name: case for name, case in exp.cases.items() if not (case.type == "hpc" and scope == "synthetic")}
    n_skipped_hpc = len(exp.cases) - len(selected)

    metadata = collect_metadata()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{timestamp}_{exp.name}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.copy2(experiment_path, run_dir / "experiment.yml")

    summary = f"{len(selected)} case{'s' if len(selected) != 1 else ''}"
    if n_skipped_hpc:
        summary += f"; skipping {n_skipped_hpc} hpc"

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

        click.echo(f"  {case_name:21s} running...", nl=False)
        try:
            if case.type == "synthetic":
                metrics = run_synthetic_case(
                    phantom_config=case.phantom,
                    recon_config=recon_config,
                    case_dir=case_dir,
                    modality=infer_modality(recon_config),
                    save_all=save_all,
                    reference_parameters=case.reference_parameters,
                    reference_metrics=case.reference_metrics,
                )
            elif case.type == "hpc":
                metrics = run_hpc_case(
                    input_path=case.input,
                    position=case.position,
                    recon_config=recon_config,
                    case_dir=case_dir,
                    save_all=save_all,
                    reference_parameters=case.reference_parameters,
                    reference_metrics=case.reference_metrics,
                    crop=case.crop,
                )

            results[case_name] = metrics

            timing = json.loads((case_dir / "timing.json").read_text())
            elapsed = timing.get("elapsed_s", 0)
            metrics["elapsed_s"] = elapsed
            click.echo("\033[2K\r", nl=False)
            _print_row(case_name, metrics, elapsed=elapsed)
        except Exception as e:
            click.echo("\033[2K\r", nl=False)
            click.echo(f"  {case_name:21s} " + click.style(f"FAILED: {e}", fg="red"))
            tb = traceback.format_exc()
            click.echo(tb, err=True)
            results[case_name] = {"error": str(e), "traceback": tb}

    # Save summary
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))


@benchmark.command()
@_output_dir_option()
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

    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        click.echo(f"  Git: {meta['git_hash']} ({meta['git_branch']})")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        results = json.loads(summary_path.read_text())
        click.echo()
        _print_summary(results)


@benchmark.command()
@_output_dir_option()
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


def _list_run_dirs(output_dir: str | Path) -> list[Path]:
    """Return all run directories under output_dir (oldest first)."""
    root = Path(output_dir)
    if not root.exists():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
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
@_output_dir_option()
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
    click.echo(f"  {'case':21s} {'metric':>10s} {'A':>10s} {'B':>10s} {'delta':>10s}")
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
            click.echo(f"  {case_name:21s} {metric:>10s} {_fmt(va)} {_fmt(vb)} {_fmt(delta)}")
            case_name = ""  # only show name on first row


@benchmark.command()
@_output_dir_option()
@click.argument("path", required=False, default=None)
@click.argument("path_b", required=False, default=None)
def view(output_dir, path, path_b):
    """Open benchmark results in napari.

    With one PATH: view that run (or case). With two PATHs: compare them
    side-by-side — raw input is loaded once from the first path, both
    reconstructions are loaded as separate layers with contrast limits
    linked between them. PATH is ``latest`` or a run name, optionally
    followed by ``/case_name``.

    \b
    Example:
      \033[92mwo bm view\033[0m
      \033[92mwo bm view latest/phase_3d_beads\033[0m
      \033[92mwo bm view RUN_A/case RUN_B/case\033[0m
    """
    click.echo(click.style("Opening benchmark results...", fg="green"))

    # Deferred imports
    import napari
    from iohub.ngff import open_ome_zarr

    from benchmarks.config import load_experiment

    output_dir = _resolve_output_dir(output_dir)

    target_a = _resolve_view_target(output_dir, path)
    if target_a is None:
        return
    run_a, case_a = target_a

    if path_b is None:
        viewer = napari.Viewer(title=f"wo bm view — {run_a.name}")
        _load_case_into_viewer(viewer, run_a, case_a, open_ome_zarr, load_experiment)
        viewer.grid.enabled = True
        napari.run()
        return

    target_b = _resolve_view_target(output_dir, path_b)
    if target_b is None:
        return
    run_b, case_b = target_b

    viewer = napari.Viewer(title=f"wo bm view — compare {run_a.name} vs {run_b.name}")

    # Raw + recon from A
    recon_layers_a = _load_case_into_viewer(viewer, run_a, case_a, open_ome_zarr, load_experiment, recon_label="a")
    # Recon from B only (raw is identical; don't duplicate)
    recon_layers_b = _load_case_into_viewer(
        viewer, run_b, case_b, open_ome_zarr, load_experiment, recon_label="b", skip_raw=True
    )

    _link_contrast_limits(recon_layers_a, recon_layers_b)

    viewer.grid.enabled = True
    napari.run()


def _resolve_view_target(output_dir: Path, path: str | None) -> tuple[Path, str | None] | None:
    """Resolve a ``path`` argument into ``(run_dir, case_name)``.

    Accepts ``None`` or ``"latest"`` for the newest run, a bare run name,
    or ``run_name/case_name``. Echoes an error and returns None if the run
    or its cases directory can't be found.
    """
    run_name = None
    case = None
    if path:
        parts = path.split("/", 1)
        run_name = parts[0]
        if len(parts) > 1:
            case = parts[1]

    if run_name in (None, "latest"):
        runs = _find_runs(output_dir, n=1)
        if not runs:
            click.echo("No benchmark runs found.")
            return None
        run_dir = runs[0]
    else:
        run_dir = output_dir / run_name
        if not run_dir.exists():
            click.echo(f"Run not found: {run_dir}")
            return None

    if not (run_dir / "cases").exists():
        click.echo(f"No cases in run {run_dir.name}.")
        return None

    return run_dir, case


def _load_case_into_viewer(
    viewer,
    run_dir: Path,
    case: str | None,
    open_ome_zarr,
    load_experiment,
    recon_label: str | None = None,
    skip_raw: bool = False,
) -> list:
    """Load raw + recon for one or more cases in ``run_dir`` into ``viewer``.

    Returns the list of napari layers added for the reconstruction(s),
    so the caller can link contrast limits between paired layers.
    """
    cases_dir = run_dir / "cases"
    exp_path = run_dir / "experiment.yml"
    experiment = load_experiment(exp_path) if exp_path.exists() else None

    case_dirs = [cases_dir / case] if case else sorted(cases_dir.iterdir())
    recon_layers = []
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        tag = f"{recon_label}/" if recon_label else ""

        if not skip_raw:
            source = _resolve_raw_source(case_dir, case_name, experiment)
            channel_filter = _input_channel_names(case_dir)
            if source and source[0].exists():
                raw_path, crop = source
                _add_zarr_to_viewer(
                    viewer,
                    raw_path,
                    f"{tag}{case_name}/raw",
                    open_ome_zarr,
                    channels=channel_filter,
                    crop=crop,
                )
            else:
                click.echo(f"  {case_name}: no raw input found")

        recon_path = case_dir / "reconstruction.zarr"
        if recon_path.exists():
            before = len(viewer.layers)
            _add_zarr_to_viewer(viewer, recon_path, f"{tag}{case_name}/recon", open_ome_zarr)
            recon_layers.extend(viewer.layers[before:])
    return recon_layers


def _link_contrast_limits(layers_a: list, layers_b: list) -> None:
    """Link contrast-limit edits between paired layers.

    Uses napari's ``link_layers`` so dragging the contrast slider on
    either side adjusts both. Also sets a common initial range.
    """
    if not layers_a or not layers_b:
        return

    from napari.experimental import link_layers

    for la, lb in zip(layers_a, layers_b):
        import numpy as np

        lo = min(float(np.asarray(la.data).min()), float(np.asarray(lb.data).min()))
        hi = max(float(np.asarray(la.data).max()), float(np.asarray(lb.data).max()))
        la.contrast_limits = (lo, hi)
        lb.contrast_limits = (lo, hi)
        link_layers([la, lb], ("contrast_limits",))


def _resolve_raw_source(case_dir: Path, case_name: str, experiment):
    """Resolve where raw data comes from for viewing.

    Returns ``(path, crop)`` where ``path`` is a zarr position path and
    ``crop`` is either None (use the whole FOV) or a CropConfig (slice
    in-memory at view time — no intermediate zarr is written). Returns
    None when no raw data can be located.
    """
    sim_path = case_dir / "simulated.zarr"
    if sim_path.exists():
        return sim_path, None
    if experiment is None or case_name not in experiment.cases:
        return None
    case_cfg = experiment.cases[case_name]
    if case_cfg.type != "hpc":
        return None
    if not case_cfg.input or not case_cfg.position:
        raise ValueError(f"hpc case '{case_name}' missing input or position in experiment.yml")
    return Path(case_cfg.input) / case_cfg.position, case_cfg.crop


def _add_zarr_to_viewer(viewer, path, prefix, open_ome_zarr, channels=None, crop=None):
    """Add channels from an OME-Zarr to the viewer.

    Handles both HCS (multi-position) and single-position (FOV) stores.
    If ``channels`` is given, only channels whose names match are added.
    If ``crop`` is given, slices Z/Y/X in-memory before adding — no
    intermediate zarr is written.
    """
    dataset = open_ome_zarr(path, mode="r")
    try:
        positions = [pos for _, pos in dataset.positions()]
    except (AttributeError, TypeError):
        positions = [dataset]

    for position in positions:
        if crop is None:
            data = np.array(position["0"][0])
        else:
            z_sl, y_sl, x_sl = crop.slices()
            data = np.array(position["0"][0, :, z_sl, y_sl, x_sl])
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
    cfg_path = case_dir / "config.yml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    names = cfg.get("input_channel_names")
    return list(names) if names else None


def _fmt(value, width=10):
    """Format a float as fixed-width scientific notation with explicit sign."""
    if value == 0:
        return f"{' 0':>{width}s}"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1e}".rjust(width)


def _sparkline(hist, n_bins=10):
    """Render a small inline sparkline (bars only)."""
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
        f"  {'case':21s} {'ref':>4s} {'time':>6s}"
        f" {'midband':>{_W}s} {'mse':>{_W}s} {'ssim':>{_W}s}"
        f"  {'min':>{_H}s} {'histogram':^12s} {'max':>{_H}s}"
    )
    click.echo(header)
    click.echo("  " + "─" * (len(header) - 2))


def _ref_badge(metrics: dict) -> str:
    """Return a 4-char ✓/✗/— badge summarising all reference_* checks.

    Shows ``—`` when no reference checks ran, ``✓`` when every check
    (parameters AND metrics) passed, and ``✗`` when any failed.
    """
    checks = [
        metrics.get("reference_parameters_check"),
        metrics.get("reference_metrics_check"),
    ]
    checks = [c for c in checks if c is not None]
    if not checks:
        return f"{'—':>4s}"
    return f"{'✓':>4s}" if all(c.get("all_pass") for c in checks) else f"{'✗':>4s}"


def _print_row(case_name: str, metrics: dict, elapsed: float | None = None):
    """Print a single summary table row."""
    iq = metrics.get("image_quality", {})
    mbp = _fmt(iq.get("midband_power", 0), _W)

    phantom = metrics.get("with_phantom", {})
    ssim_str = f"{phantom['ssim']:>{_W}.3f}" if "ssim" in phantom else f"{'—':>{_W}s}"
    mse_str = _fmt(phantom["mse"], _W) if "mse" in phantom else f"{'—':>{_W}s}"
    ref_str = _ref_badge(metrics)

    hist = iq.get("histogram", {})
    if hist:
        edges = np.array(hist["bin_edges"])
        min_str = _fmt(edges[0], _H)
        max_str = _fmt(edges[-1], _H)
        spark = _sparkline(hist)
    else:
        min_str = f"{'—':>{_H}s}"
        max_str = f"{'—':>{_H}s}"
        spark = ""

    time_str = f"{elapsed:5.1f}s" if elapsed is not None else f"{'—':>6s}"

    click.echo(f"  {case_name:21s} {ref_str} {time_str} {mbp} {mse_str} {ssim_str}  {min_str} {spark:^12s} {max_str}")


def _print_summary(results: dict):
    """Print a full summary table (header + all rows)."""
    _print_header()
    for case_name, metrics in results.items():
        if "error" in metrics:
            click.echo(f"  {case_name:21s} " + click.style("ERROR", fg="red"))
            continue
        _print_row(case_name, metrics, elapsed=metrics.get("elapsed_s"))
