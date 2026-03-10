# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Waveorder is a framework for computational microscopy, providing wave-optical simulations and reconstruction algorithms for label-free imaging (QLIPP, PTI, QPI, DPC) and fluorescence microscopy (widefield deconvolution, oblique plane light-sheet). Built on PyTorch for GPU acceleration, xarray/zarr for scientific data, and Click for CLI.

**Python ≥3.11 required.**

## Common Commands

```bash
# Environment setup (uses uv)
uv sync --group dev --extra visual

# Run all tests
uv run pytest

# Run a single test file or specific test
uv run pytest tests/api_tests/test_api_examples.py
uv run pytest tests/api_tests/test_api_examples.py::test_function_name -v

# Linting and formatting (ruff, line length 120)
uvx ruff check .
uvx ruff format .
uvx ruff check . --fix

# Pre-commit hooks
pre-commit run --all-files

# Build docs
cd docs/ && uv run sphinx-build -M html ./ ./build

# CLI entry points: `waveorder` or `wo`
uv run waveorder --help
```

## Architecture

### Data Flow

```
Input (OME-Zarr) → Settings (Pydantic) → API Layer → Transfer Function → Inverse Reconstruction → Output (xarray.DataArray → OME-Zarr)
```

### Key Layers

- **`waveorder/api/`** — Main user-facing computational interface. Each module (`phase.py`, `birefringence.py`, `fluorescence.py`) follows a consistent pattern: `simulate()` → `compute_transfer_function()` → `apply_inverse_transfer_function()`, or the one-liner `reconstruct()`. Settings are Pydantic models in `_settings.py`.

- **`waveorder/models/`** — Low-level wave-optical simulation models (e.g., `phase_thick_3d.py`, `isotropic_thin_3d.py`). Each model provides `generate_test_phantom()`, `calculate_transfer_function()`, and `apply_transfer_function()`. These are the physics engines called by the API layer.

- **`waveorder/cli/`** — Click-based CLI (`waveorder`/`wo`). Commands: `simulate` (`sim`), `reconstruct` (`rec`), `compute-transfer-function` (`compute-tf`), `apply-inverse-transfer-function` (`apply-inv-tf`), `view` (`v`), `interactive` (`gui`). Uses `AliasGroup` for short aliases. Config YAML → Pydantic model via `yaml_to_model()` → API call → zarr output.

- **`waveorder/plugin/`** — napari plugin and Qt GUI (`gui.py`, `main_widget.py`, `tab_recon.py`).

- **Core modules** — `optics.py` (FFT-based field computations), `filter.py`, `focus.py`, `correction.py`, `background_estimator.py`.

### Settings/Configuration Pattern

Settings use Pydantic v2 with `ConfigDict(extra="forbid")`. Three-tier hierarchy: `MyBaseModel` → `FourierTransferFunctionSettings` → API-specific settings. YAML config files are validated through these models. Cross-field validation uses `model_validator` and `field_validator` decorators.

### Data Format

OME-Zarr standard via iohub. Data represented as `xarray.DataArray` with CZYX dimensions. HCS layout for multi-position, FOV layout for single position.

## Testing

Tests are in `tests/` organized by layer: `api_tests/`, `cli_tests/`, `models/`, `widget_tests/`, `calibration_tests/`. Key fixtures in `conftest.py` include `make_czyx()` (synthetic data factory) and `example_plate()` (HCS plate dataset). Uses hypothesis for property-based testing and pytest-qt for GUI tests.

## Notes

- iohub dependency is installed from git (`czbiohub-sf/iohub@xarray-integration`)
- M1/Mac GPU: Set `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Ruff config: double quotes, space indent, line length 120
