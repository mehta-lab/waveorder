"""Imaging-system calibration: parameter vectors, PSF/OTF synthesis, scores.

This package holds the numerics behind ``calibrate-tf``:

* :mod:`waveorder.calibrate.zernike_names` — named Zernike terms
  (``defocus`` ... ``spherical``, ``z<n>``) to Noll indices.
* :mod:`waveorder.calibrate.synthesis` — differentiable fluorescence
  PSF/OTF synthesis from a parameter vector (Zernike coefficients in waves).
* :mod:`waveorder.calibrate.scores` — Whittle/Gaussian evidence and
  related calibration scores.
"""

from waveorder.calibrate.scores import (
    calibration_penalty,
    whittle_evidence,
)
from waveorder.calibrate.synthesis import (
    fluorescence_otf,
    fluorescence_psf,
)
from waveorder.calibrate.zernike_names import (
    NAMED_TERMS,
    name_to_noll,
    noll_indices_up_to_order,
    noll_to_name,
    validate_zernike_name,
)

__all__ = [
    "NAMED_TERMS",
    "calibration_penalty",
    "fluorescence_otf",
    "fluorescence_psf",
    "name_to_noll",
    "noll_indices_up_to_order",
    "noll_to_name",
    "validate_zernike_name",
    "whittle_evidence",
]
