"""Backward-compatibility re-export from waveorder.projection.

All Siddon ray-tracing, ramp filter, and CG-Tikhonov code now lives in
``waveorder.projection``. This shim preserves ``from siddon import ...``
in scripts that have not yet been updated.
"""

from waveorder.projection import (  # noqa: F401
    SiddonOperator,
    cg_tikhonov,
    ramp_filter_sinogram,
    siddon_backproject,
    siddon_project,
)
