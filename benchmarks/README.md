# WaveOrder Benchmarks

Tracks reconstruction quality and performance. Synthetic cases
generate a phantom, simulate a measurement, reconstruct via `wo rec`, and
score against ground truth. Real cases require access to the Biohub HPC and
reconstruct existing data.

## Install (from scratch)

    git clone https://github.com/mehta-lab/waveorder && cd waveorder
    uv sync --group bench
    source .venv/bin/activate  # or prepend `uv run` to commands

`wo bm` / `wo benchmark` appears once the `bench` group is installed
and the venv is active.

## First run

    export WAVEORDER_BENCH_OUTPUT=/hpc/projects/waveorder/benchmarks  # on biohub HPC
    wo bm run
    wo bm latest

## Commands

    wo bm run [--scope synthetic|all] [-e EXPERIMENT] [--save-all]
        Run cases. Default scope is synthetic (skips hpc cases).

    wo bm latest
        Summary table (timing, metrics, histograms) for the newest run.

    wo bm history [-n N]
        List recent run directories.

    wo bm compare [RUN_A RUN_B]
        Diff metrics between two runs (default: the two newest).

    wo bm view [latest|RUN_NAME][/CASE_NAME]
        Open raw input + reconstruction side-by-side in napari grid mode.

## Adding a benchmark case

1. Add a reconstruction config to `benchmarks/configs/`.
2. Add a case to `benchmarks/experiments/regression.yml`:

        my_case:
          type: synthetic
          phantom:
            function: single_bead
            shape: [64, 128, 128]
            pixel_sizes: [0.25, 0.1, 0.1]
            bead_radius_um: 2.5
          config: ../configs/my_case.yml

   For HPC: `type: hpc`, plus `input` (path to an OME-Zarr) and `position`.

3. `wo bm run --scope synthetic` to run it.

## Custom experiments

Point `-e` at any experiment YAML:

    wo bm run -e my_sweep.yml

Cases can share a `base_phantom` and `base_config` and apply per-case
`overrides` via dotted keys. See `benchmarks/experiments/regression.yml`
for the registered suite.

## Output layout

    <output_root>/
    └── <timestamp>_<experiment>/
        ├── metadata.json              # git hash, branch, gpu, versions
        ├── experiment.yml             # copy of the experiment input
        ├── summary.json               # per-case metrics
        └── cases/<case>/
            ├── config.yml             # recon config (after overrides)
            ├── phantom_config.json    # synthetic only — phantom params
            ├── phantom.zarr           # synthetic only — ground truth
            ├── simulated.zarr         # synthetic only — forward model
            ├── reconstruction.zarr    # `wo rec` output
            ├── cli_command.sh         # the exact `wo rec` call
            ├── timing.json
            └── metrics.json
