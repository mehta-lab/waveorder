"""Named Zernike terms and their Noll indices.

Convention (fixed for WaveOrder 4.0, see the design document section 4.4):

* Indexing is Noll (``j = 1`` is piston), evaluated by
  :func:`waveorder.zernike.zernike_noll` with the orthonormal
  normalization, so a coefficient is the RMS wavefront error of its term.
* Coefficients are in **waves** of pupil phase: the pupil is
  ``P0 * exp(i 2 pi sum_j c_j Z_j)``.

  .. warning:: UNIT CONVERSION. Some collaborating pipelines (e.g. the
     blind-deconvolution track) exchange Zernike coefficients in
     **radians** of pupil phase: ``c_radians = 2 * pi * c_waves``. Every
     function in :mod:`waveorder.calibrate` takes and returns **waves**;
     convert explicitly at the boundary and never mix the two silently.
* The pupil azimuth ``theta = atan2(f_y, f_x)`` is zero along the +x
  frequency axis, matching :func:`waveorder.zernike.make_pupil_grid`.
* Named terms: ``defocus`` (Noll 4), ``astigmatism_45`` (5, the
  ``sin 2theta`` term), ``astigmatism_0`` (6, ``cos 2theta``), ``coma_y``
  (7, ``sin theta``), ``coma_x`` (8, ``cos theta``), ``trefoil_y`` (9,
  ``sin 3theta``), ``trefoil_x`` (10, ``cos 3theta``), ``spherical`` (11).
  Any other term is addressed as ``z<n>`` by Noll index, e.g. ``z12``.
* ``zernike_up_to_order: {order: n}`` frees every Noll index through
  radial order ``n`` **excluding piston, tip and tilt** (Noll 1-3): piston
  does not change an intensity PSF, and tip/tilt are degenerate with a
  lateral translation of the object.
"""

from __future__ import annotations

import re

from waveorder.zernike import _noll_to_nm

NAMED_TERMS: dict[str, int] = {
    "defocus": 4,
    "astigmatism_45": 5,
    "astigmatism_0": 6,
    "coma_y": 7,
    "coma_x": 8,
    "trefoil_y": 9,
    "trefoil_x": 10,
    "spherical": 11,
}

_NOLL_TO_NAME: dict[int, str] = {j: name for name, j in NAMED_TERMS.items()}

_ZN_PATTERN = re.compile(r"^z([0-9]+)$")


def name_to_noll(name: str) -> int:
    """Convert a named Zernike term to its Noll index.

    Parameters
    ----------
    name : str
        A named term (``defocus``, ``astigmatism_0``, ...) or ``z<n>`` by
        Noll index.

    Returns
    -------
    int
        Noll index (1-based).

    Raises
    ------
    ValueError
        If the name is not a recognized term.
    """
    if name in NAMED_TERMS:
        return NAMED_TERMS[name]
    match = _ZN_PATTERN.match(name)
    if match:
        j = int(match.group(1))
        if j < 1:
            raise ValueError(f"Noll index must be >= 1, got {name!r}")
        return j
    raise ValueError(f"Unknown Zernike term {name!r}. Use one of {sorted(NAMED_TERMS)} or 'z<n>' with a Noll index n.")


def noll_to_name(j: int) -> str:
    """Canonical name of a Noll index: the named term if it has one, else ``z<n>``."""
    return _NOLL_TO_NAME.get(j, f"z{j}")


def validate_zernike_name(name: str) -> str:
    """Validate a Zernike term name, returning it unchanged.

    Raises
    ------
    ValueError
        If the name is neither a named term nor ``z<n>``.
    """
    name_to_noll(name)
    return name


def noll_indices_up_to_order(order: int) -> list[int]:
    """Noll indices of every term through radial order ``order``.

    Piston, tip and tilt (Noll 1-3) are excluded; see the module docstring.

    Parameters
    ----------
    order : int
        Highest radial order ``n`` to include, ``order >= 2``.

    Returns
    -------
    list[int]
        Sorted Noll indices with ``4 <= j`` and radial order ``<= order``.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    indices = []
    j = 4
    while True:
        n, _ = _noll_to_nm(j)
        if n > order:
            break
        indices.append(j)
        j += 1
    return indices
