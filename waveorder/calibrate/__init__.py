"""Imaging-system calibration: parameter vectors, PSF/OTF synthesis, scores.

This package holds the numerics behind ``calibrate-tf``:

* :mod:`waveorder.calibrate.zernike_names` — named Zernike terms
  (``defocus`` ... ``spherical``, ``z<n>``) to Noll indices.
* :mod:`waveorder.calibrate.synthesis` — differentiable fluorescence
  PSF/OTF synthesis from a parameter vector (Zernike coefficients in waves).
* :mod:`waveorder.calibrate.scores` — Whittle/Gaussian evidence and
  related calibration scores.
* :mod:`waveorder.calibrate.tfzarr` — the ``tf.zarr`` store.
* :mod:`waveorder.calibrate.run`, :mod:`waveorder.calibrate.evidence_search`
  — configuration-driven searches (loaded lazily: they depend on
  :mod:`waveorder.config`, which itself imports the leaf modules above, so
  importing them here eagerly would be circular).
"""

import importlib

from waveorder.calibrate.scores import (
    calibration_penalty,
    whittle_evidence,
)
from waveorder.calibrate.synthesis import (
    fluorescence_otf,
    fluorescence_psf,
)
from waveorder.calibrate.tfzarr import (
    TransferFunctionStore,
    read_tf_zarr,
    write_tf_zarr,
)
from waveorder.calibrate.zernike_names import (
    NAMED_TERMS,
    name_to_noll,
    noll_indices_up_to_order,
    noll_to_name,
    validate_zernike_name,
)

_LAZY_EXPORTS = {
    "CalibrationResult": "waveorder.calibrate.run",
    "build_recovery_settings": "waveorder.calibrate.run",
    "run_zernike_calibration": "waveorder.calibrate.run",
    "zernike_calibration_requested": "waveorder.calibrate.run",
    "EvidenceCalibrationResult": "waveorder.calibrate.evidence_search",
    "gcv_score": "waveorder.calibrate.evidence_search",
    "run_evidence_calibration": "waveorder.calibrate.evidence_search",
}

__all__ = [
    "NAMED_TERMS",
    "CalibrationResult",
    "EvidenceCalibrationResult",
    "TransferFunctionStore",
    "build_recovery_settings",
    "calibration_penalty",
    "fluorescence_otf",
    "fluorescence_psf",
    "gcv_score",
    "name_to_noll",
    "noll_indices_up_to_order",
    "noll_to_name",
    "read_tf_zarr",
    "run_evidence_calibration",
    "run_zernike_calibration",
    "validate_zernike_name",
    "whittle_evidence",
    "write_tf_zarr",
    "zernike_calibration_requested",
]


def __getattr__(name: str):
    """Load the config-dependent calibration entry points on first use."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
