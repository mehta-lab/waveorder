"""Pydantic models for the WaveOrder 4.0 configuration schema.

The schema follows the WaveOrder 4.0 design document: five blocks (``data``,
``imaging_system``, ``object``, ``noise``, ``solver``) plus ``outputs`` and a
required ``version`` field.

Conventions
-----------
* Any number under ``imaging_system`` may be written as
  ``{nominal: x, tolerance: t}`` (a :class:`FreeParameter`), which makes it a
  free parameter of the imaging system, estimated from the data. ``tolerance``
  is the standard deviation of a Gaussian prior in the parameter's own units,
  or ``"free"`` for a flat prior. Optional ``bounds: [min, max]`` truncate the
  prior. A plain number is fixed.
* Where a field offers several options, its value is a bare name for an
  option with no parameters, or a mapping with one key (the option's name)
  holding its parameters, e.g. ``reconstruction: closed_form`` or
  ``reconstruction: {fista: {iterations: 50}}``.
* Zernike coefficients under ``pupil_aberrations`` are named (``defocus``,
  ``astigmatism_0``, ``astigmatism_45``, ``coma_x``, ``coma_y``,
  ``trefoil_x``, ``trefoil_y``, ``spherical``, then ``z<n>`` by Noll index)
  and are in waves of RMS wavefront error (orthonormal Noll normalization).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from waveorder._pixel_size import YXPixelSize
from waveorder.calibrate.zernike_names import noll_indices_up_to_order, validate_zernike_name


class V4BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Free parameters -------------------------------------------------------


class FreeParameter(V4BaseModel):
    """A calibratable imaging-system number: ``{nominal, tolerance, bounds}``.

    Parameters
    ----------
    nominal : float
        Center of the prior and the fixed value when nothing is estimated.
    tolerance : float or "free"
        Standard deviation of a Gaussian prior centred on ``nominal``, in the
        parameter's own units, or ``"free"`` for a flat prior.
    bounds : tuple[float, float], optional
        Truncates the prior; the solver clamps at these bounds.
    """

    nominal: float
    tolerance: Union[PositiveFloat, Literal["free"]] = "free"
    bounds: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _nominal_inside_bounds(self):
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo >= hi:
                raise ValueError(f"bounds must satisfy min < max, got {self.bounds}")
            if not (lo <= self.nominal <= hi):
                raise ValueError(f"nominal = {self.nominal} lies outside bounds = {self.bounds}")
        return self


CalibratableFloat = Union[float, FreeParameter]


def nominal_of(value: CalibratableFloat | None) -> float | None:
    """Return the nominal value of a plain number or a :class:`FreeParameter`."""
    if value is None:
        return None
    if isinstance(value, FreeParameter):
        return value.nominal
    return float(value)


def is_free(value: Any) -> bool:
    """Whether a value is a :class:`FreeParameter` (i.e. carries a tolerance)."""
    return isinstance(value, FreeParameter)


# --- data ------------------------------------------------------------------


class PixelSize(V4BaseModel):
    """Voxel sizes in micrometers."""

    yx: Union[PositiveFloat, YXPixelSize] = Field(
        description="lateral pixel size in um; scalar for isotropic, or {y, x} mapping",
    )
    z: PositiveFloat = Field(description="axial pixel size in um")

    @field_validator("yx", mode="before")
    @classmethod
    def _normalize_yx(cls, v):
        return YXPixelSize.from_value(v)


class DataSettings(V4BaseModel):
    """The ``data`` block: which data *g* is reconstructed and its sampling."""

    input_channel_names: list[str] = Field(
        default_factory=lambda: ["simulated"],
        description="channel names in the input zarr (defaults only make sense for simulate)",
    )
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = Field(
        default="all", description="time points to reconstruct"
    )
    pixel_size: PixelSize


# --- imaging_system --------------------------------------------------------


class ZernikeUpToOrder(V4BaseModel):
    """Free every Zernike term through radial order ``order`` with nominal 0.

    Piston, tip and tilt (Noll 1-3) are excluded: piston does not affect an
    intensity PSF and tip/tilt are degenerate with an object translation.
    Named entries in :class:`PupilAberrations` override this shorthand for
    their term.
    """

    order: PositiveInt = Field(ge=2, description="highest radial order to free")
    tolerance: Union[PositiveFloat, Literal["free"]] = Field(
        default="free", description="Gaussian prior std in waves, or 'free'"
    )
    bounds: Optional[tuple[float, float]] = None


_CALIBRATABLE_ADAPTER = TypeAdapter(CalibratableFloat)


class PupilAberrations(BaseModel):
    """Zernike pupil aberrations by name; unlisted terms are 0 and fixed.

    Accepts a ``zernike_up_to_order`` shorthand plus any number of named
    terms (``defocus``, ``astigmatism_0``, ``astigmatism_45``, ``coma_x``,
    ``coma_y``, ``trefoil_x``, ``trefoil_y``, ``spherical``, ``z<n>`` by
    Noll index). Coefficients are in waves (RMS, orthonormal Noll).
    """

    model_config = ConfigDict(extra="allow")

    zernike_up_to_order: Optional[ZernikeUpToOrder] = None

    @model_validator(mode="after")
    def _validate_terms(self):
        extra = self.__pydantic_extra__ or {}
        for name, value in extra.items():
            validate_zernike_name(name)
            extra[name] = _CALIBRATABLE_ADAPTER.validate_python(value)
        return self

    @property
    def terms(self) -> dict[str, CalibratableFloat]:
        """Named Zernike terms, validated, excluding the shorthand."""
        return dict(self.__pydantic_extra__ or {})


class Pinhole(V4BaseModel):
    """Confocal pinhole with a calibratable diameter.

    Exactly one of ``diameter_au`` (Airy units) or ``diameter_um``
    (micrometers in demagnified image space, the 3.x convention) must be set.
    """

    diameter_au: Optional[CalibratableFloat] = None
    diameter_um: Optional[CalibratableFloat] = None

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.diameter_au is None) == (self.diameter_um is None):
            raise ValueError("pinhole requires exactly one of 'diameter_au' or 'diameter_um'")
        return self


class Detection(V4BaseModel):
    """Detection arm of the imaging system."""

    wavelength: Optional[PositiveFloat] = Field(
        default=None, description="emission wavelength in um (fluorescence only)"
    )
    numerical_aperture: CalibratableFloat = Field(default=1.2, description="detection objective numerical aperture")
    pinhole: Union[Literal["none", "point"], Pinhole] = Field(
        default="none", description="none (widefield) | point | {diameter_au | diameter_um}"
    )
    pupil_aberrations: Optional[PupilAberrations] = None
    z_focus_offset: CalibratableFloat = Field(default=0.0, description="offset from center slice in slice units")

    @model_validator(mode="after")
    def _positive_na(self):
        na = nominal_of(self.numerical_aperture)
        if na is not None and na <= 0:
            raise ValueError(f"detection.numerical_aperture must be positive, got {na}")
        return self


class Tilt(V4BaseModel):
    """Illumination tilt in radians."""

    zenith: CalibratableFloat = 0.0
    azimuth: CalibratableFloat = 0.0


class Illumination(V4BaseModel):
    """Illumination arm: transillumination for phase, excitation for confocal."""

    wavelength: Optional[PositiveFloat] = Field(default=None, description="illumination wavelength in um")
    numerical_aperture: Optional[CalibratableFloat] = Field(
        default=None, description="condenser / excitation numerical aperture"
    )
    tilt: Tilt = Field(default_factory=Tilt)
    invert_phase_contrast: bool = Field(default=False, description="phase only")


class TiledFieldDependence(V4BaseModel):
    """Tiled shift-variance: one parameter set per tile, blended in overlaps."""

    size_yx: tuple[PositiveInt, PositiveInt] = Field(description="tile size in pixels (y, x)")
    overlap_yx: tuple[NonNegativeInt, NonNegativeInt] = Field(default=(32, 32), description="tile overlap in pixels")
    tile_blend: Literal["linear", "cosine"] = "linear"


class FieldDependenceTiled(V4BaseModel):
    tiled: TiledFieldDependence


class ImagingSystem(V4BaseModel):
    """The ``imaging_system`` block: H and its prior (via tolerances)."""

    contrast: Literal["phase", "fluorescence"]
    field_dependence: Union[Literal["shift_invariant"], FieldDependenceTiled] = "shift_invariant"
    index_of_refraction_media: PositiveFloat = 1.3
    detection: Detection = Field(default_factory=Detection)
    illumination: Illumination = Field(default_factory=Illumination)

    @model_validator(mode="after")
    def _na_below_index(self):
        na_det = nominal_of(self.detection.numerical_aperture)
        if na_det is not None and na_det > self.index_of_refraction_media:
            raise ValueError(
                f"detection.numerical_aperture = {na_det} must be less than or equal to "
                f"index_of_refraction_media = {self.index_of_refraction_media}"
            )
        na_ill = nominal_of(self.illumination.numerical_aperture)
        if na_ill is not None and na_ill > self.index_of_refraction_media:
            raise ValueError(
                f"illumination.numerical_aperture = {na_ill} must be less than or equal to "
                f"index_of_refraction_media = {self.index_of_refraction_media}"
            )
        return self


# --- object ----------------------------------------------------------------


class TikhonovSpectralPrior(V4BaseModel):
    exponent: Optional[float] = Field(default=None, description="power-law exponent of the object spectrum")
    fit_from: Optional[str] = Field(default=None, description="path to fit the spectrum from")


class BeadsPrior(V4BaseModel):
    diameter: PositiveFloat = Field(description="bead diameter in um; sparse spheres at unknown positions")
    snr_threshold: PositiveFloat = Field(
        default=6.0,
        description="initial detection threshold in robust noise units (median + snr_threshold * 1.4826 * MAD); "
        "detection relaxes adaptively down to half this value before failing",
    )
    max_beads: PositiveInt = Field(default=50, description="fit at most this many beads, brightest first")
    min_separation: PositiveInt = Field(default=5, description="non-maximum suppression window in voxels")


class DenoiserPrior(V4BaseModel):
    checkpoint: str


class PriorTikhonovSpectral(V4BaseModel):
    tikhonov_spectral: TikhonovSpectralPrior


class PriorBeads(V4BaseModel):
    beads: BeadsPrior


class PriorDenoiser(V4BaseModel):
    denoiser: DenoiserPrior


Prior = Union[
    Literal["tikhonov", "tv", "none"],
    PriorTikhonovSpectral,
    PriorBeads,
    PriorDenoiser,
]


class LegacyAutoWeight(V4BaseModel):
    """The 3.x ``auto_regularization`` sweep, carried losslessly through migration.

    The bare ``weight: auto`` of the 4.0 design (score-based selection of
    lambda) is a separate option; this block keeps the 3.x rule-based sweep
    (``otsu_cnr`` / ``l_curve``) available to migrated configurations.
    """

    rule: Literal["otsu_cnr", "l_curve"] = "otsu_cnr"
    search_min: float = -6.0
    search_max: float = 2.0
    num_samples: PositiveInt = 25
    search_scale: Literal["relative", "absolute"] = "relative"
    report_path: Optional[str] = None
    plot_path: Optional[str] = None


class WeightAuto(V4BaseModel):
    auto: LegacyAutoWeight


class GaussianSpectrumSource(V4BaseModel):
    """Gaussian random field with a power-law spectrum (simulation only)."""

    exponent: float = Field(default=2.0, description="power-spectrum decay exponent")


class PhantomSource(V4BaseModel):
    kind: str


class FromReconSource(V4BaseModel):
    path: str


class SourceGaussianSpectrum(V4BaseModel):
    gaussian_spectrum: GaussianSpectrumSource


class SourcePhantom(V4BaseModel):
    phantom: PhantomSource


class SourceFromRecon(V4BaseModel):
    from_recon: FromReconSource


Source = Union[SourceGaussianSpectrum, SourcePhantom, SourceFromRecon]


class ObjectSettings(V4BaseModel):
    """The ``object`` block: the prior on *f* (or its ``source`` in a simulation)."""

    dimension: Literal[2, 3] = Field(default=3, description="3 | 2 (thin object, a single plane)")
    prior: Prior = "tikhonov"
    weight: Union[NonNegativeFloat, Literal["auto"], WeightAuto] = Field(
        default=1e-3, description="regularization weight lambda, or auto"
    )
    constraints: list[Literal["nonneg"]] = Field(default_factory=list)
    source: Optional[Source] = Field(
        default=None, description="the true object of a simulation configuration (design section 5)"
    )


# --- noise -----------------------------------------------------------------


class DarkFrames(V4BaseModel):
    dark_frames: str


class GaussianNoise(V4BaseModel):
    sigma: Union[PositiveFloat, Literal["estimate"], DarkFrames] = 1.0
    snr_db: Optional[float] = Field(
        default=None, description="simulation only: sets sigma from the clean signal's power"
    )


class PoissonNoise(V4BaseModel):
    gain: PositiveFloat = 1.0
    offset: float = 0.0
    photons: Optional[PositiveFloat] = Field(
        default=None, description="simulation only: mean photons per voxel of the clean signal"
    )


class PoissonGaussianNoise(V4BaseModel):
    gain: PositiveFloat = 1.0
    offset: float = 0.0
    sigma: Union[PositiveFloat, Literal["estimate"], DarkFrames] = 1.0


class NoiseModel(V4BaseModel):
    """The ``noise`` block: exactly one of gaussian / poisson / poisson_gaussian."""

    gaussian: Optional[GaussianNoise] = None
    poisson: Optional[PoissonNoise] = None
    poisson_gaussian: Optional[PoissonGaussianNoise] = None

    @model_validator(mode="after")
    def _exactly_one(self):
        set_options = [name for name in ("gaussian", "poisson", "poisson_gaussian") if getattr(self, name) is not None]
        if len(set_options) == 0:
            self.gaussian = GaussianNoise()
        elif len(set_options) > 1:
            raise ValueError(f"noise accepts exactly one option, got {set_options}")
        return self

    @property
    def kind(self) -> str:
        for name in ("gaussian", "poisson", "poisson_gaussian"):
            if getattr(self, name) is not None:
                return name
        return "gaussian"


# --- solver ----------------------------------------------------------------


class Padding(V4BaseModel):
    # Default 0 preserves 3.x behavior exactly (3.x z_padding defaults to 0).
    # The design document's section 2 text says "padding: {z: 5, yx: 0}",
    # which contradicts its own defaults-reproduce-3.x requirement; the
    # deviation is documented in PR #592.
    z: NonNegativeInt = Field(default=0, description="z slices to pad for axial boundary effects")
    yx: Union[NonNegativeInt, tuple[NonNegativeInt, NonNegativeInt]] = Field(
        default=0, description="transverse padding in pixels; number or [y, x]"
    )


class FistaReconstruction(V4BaseModel):
    iterations: PositiveInt = 50


class AdmmReconstruction(V4BaseModel):
    iterations: PositiveInt = 1
    rho: PositiveFloat = 1e-3


class RichardsonLucyReconstruction(V4BaseModel):
    """Richardson-Lucy options, matching the 3.x ``rl`` block field for field."""

    iterations: PositiveInt = 25
    background: NonNegativeFloat = 0.0
    stopping_tolerance: Optional[NonNegativeFloat] = None
    back_projector: Literal["matched", "gaussian", "butterworth", "wiener", "wiener_butterworth"] = "matched"
    bp_alpha: Optional[PositiveFloat] = None
    bp_beta: Optional[PositiveFloat] = None
    gradient_consensus: bool = Field(
        default=False, description="use the RLGC gradient-consensus stopping variant (3.x 'RLGC')"
    )


class ReconstructionFista(V4BaseModel):
    fista: FistaReconstruction


class ReconstructionAdmm(V4BaseModel):
    admm: AdmmReconstruction


class ReconstructionRichardsonLucy(V4BaseModel):
    richardson_lucy: RichardsonLucyReconstruction


Reconstruction = Union[
    Literal["closed_form"],
    ReconstructionFista,
    ReconstructionAdmm,
    ReconstructionRichardsonLucy,
]


class GridStage(V4BaseModel):
    points: Union[PositiveInt, dict[str, PositiveInt]] = 7


class RandomStage(V4BaseModel):
    samples: PositiveInt = 16
    seed: int = 0


class GradientStage(V4BaseModel):
    max_iter: PositiveInt = 200
    method: Literal["split", "alternating", "joint"] = "split"


class SearchGrid(V4BaseModel):
    grid: GridStage


class SearchRandom(V4BaseModel):
    random: RandomStage


class SearchGradient(V4BaseModel):
    gradient: GradientStage


SearchStage = Union[SearchGrid, SearchRandom, SearchGradient, Literal["grid", "random", "gradient"]]


def _normalize_search_stage(stage: SearchStage) -> Union[SearchGrid, SearchRandom, SearchGradient]:
    """Expand a bare stage name into its option model with defaults."""
    if stage == "grid":
        return SearchGrid(grid=GridStage())
    if stage == "random":
        return SearchRandom(random=RandomStage())
    if stage == "gradient":
        return SearchGradient(gradient=GradientStage())
    return stage


class CalibrationAdvanced(V4BaseModel):
    """Step sizes and other knobs of the search; carries 3.x optimizer settings."""

    optimizer: Optional[Literal["adam", "nadam", "lbfgs", "nelder_mead", "grid_search"]] = Field(
        default=None, description="3.x optimizer backend; null = not carried from a 3.x optimization block"
    )
    learning_rates: dict[str, float] = Field(
        default_factory=dict,
        description="per-parameter gradient step sizes, keyed by dotted parameter name",
    )
    convergence_tol: Optional[float] = None
    convergence_patience: Optional[PositiveInt] = 5
    use_gradients: Optional[bool] = None
    grid_points: PositiveInt = 7
    loss: Optional[dict[str, Any]] = Field(default=None, description="3.x sharpness loss settings (type + parameters)")
    log_dir: Optional[str] = None


class Calibration(V4BaseModel):
    """How H is estimated; read only when something has a tolerance."""

    score: Literal["evidence", "gcv", "sure", "sharpness"] = "evidence"
    search: list[SearchStage] = Field(default_factory=lambda: [SearchGradient(gradient=GradientStage())])
    final_estimate: Literal["best_H", "average_near_best_H"] = "best_H"
    advanced: Optional[CalibrationAdvanced] = None

    @field_validator("search", mode="after")
    @classmethod
    def _normalize_stages(cls, v):
        return [_normalize_search_stage(stage) for stage in v]


class SolverSettings(V4BaseModel):
    """The ``solver`` block: numerics of the inner and outer problems."""

    padding: Padding = Field(default_factory=Padding)
    reconstruction: Reconstruction = "closed_form"
    calibration: Optional[Calibration] = None
    apodization_rolloff: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="raised-cosine roll-off of the inverse filter at the transverse Nyquist edge",
    )
    device: Optional[str] = Field(default=None, description="compute device: null=cpu, auto, cuda:0, mps, ...")


# --- outputs ---------------------------------------------------------------


OutputName = Literal[
    "reconstruction",
    "transfer_function",
    "psf",
    "parameters",
    "parameters_covariance",
    "score_trace",
    "score_grid",
    "weight_trace",
    "residual",
    "tile_map",
]

_DEFAULT_OUTPUTS: tuple[str, ...] = ("reconstruction", "transfer_function")


# --- top level --------------------------------------------------------------


class SimulateSettings(V4BaseModel):
    """The ``simulate`` block of a simulation configuration (design section 5)."""

    seed: int = 0
    shape: tuple[PositiveInt, PositiveInt, PositiveInt]


class V4Config(V4BaseModel):
    """A complete WaveOrder 4.0 configuration."""

    version: Literal[4]
    data: DataSettings
    imaging_system: ImagingSystem
    object: ObjectSettings = Field(default_factory=ObjectSettings)
    noise: NoiseModel = Field(default_factory=NoiseModel)
    solver: SolverSettings = Field(default_factory=SolverSettings)
    outputs: Union[Literal["all"], list[OutputName]] = Field(default_factory=lambda: list(_DEFAULT_OUTPUTS))
    simulate: Optional[SimulateSettings] = None

    @model_validator(mode="after")
    def _validation_rules(self):
        """The cross-block validation rules of design section 4.5."""
        params = free_parameters(self.imaging_system)
        anything_estimated = len(params) > 0
        calibration = self.solver.calibration

        if isinstance(self.object.prior, PriorBeads) and not anything_estimated:
            raise ValueError("object.prior: beads requires a free imaging-system parameter (beads are for calibration)")

        if calibration is not None and anything_estimated:
            if (
                calibration.score == "sharpness"
                and self.imaging_system.field_dependence != "shift_invariant"
                and all(p.tolerance == "free" for p in params.values())
            ):
                warnings.warn(
                    "tiled field dependence with every tolerance 'free' and score: sharpness "
                    "estimates each tile independently with no prior; expect overfitting on "
                    "low-signal tiles.",
                    UserWarning,
                )
            if calibration.score == "evidence":
                prior_ok = self.object.prior in ("tikhonov",) or isinstance(
                    self.object.prior, (PriorTikhonovSpectral, PriorBeads)
                )
                if not prior_ok:
                    raise ValueError("score: evidence requires object.prior in {tikhonov, tikhonov_spectral, beads}")
                if self.noise.kind == "poisson":
                    raise ValueError("score: evidence requires Gaussian-family noise")
            for stage in calibration.search:
                if isinstance(stage, (SearchGrid, SearchRandom)):
                    unbounded = [name for name, p in params.items() if p.tolerance == "free" and p.bounds is None]
                    if unbounded:
                        raise ValueError(
                            f"a grid or random search stage requires bounds for parameters with a "
                            f"'free' tolerance; missing bounds on: {unbounded}"
                        )
                if isinstance(stage, SearchGrid):
                    points = stage.grid.points
                    count = 1
                    for name in params:
                        n = points if isinstance(points, int) else points.get(name, 7)
                        if name.endswith(".zernike_up_to_order"):
                            shorthand = self.imaging_system.detection.pupil_aberrations.zernike_up_to_order
                            count *= n ** len(noll_indices_up_to_order(shorthand.order))
                        else:
                            count *= n
                    if count > 10_000:
                        warnings.warn(
                            f"the grid search stage evaluates {count} points over "
                            f"{len(params)} parameters; consider fewer points per "
                            f"parameter or a random stage.",
                            UserWarning,
                        )
                if (
                    isinstance(stage, SearchGradient)
                    and stage.gradient.method == "joint"
                    and calibration.score == "sharpness"
                ):
                    warnings.warn(
                        "gradient method 'joint' with score 'sharpness' collapses to the trivial "
                        "solution; use 'split' or 'alternating', or an evidence-family score.",
                        UserWarning,
                    )

        if self.object.weight == "auto":
            score = calibration.score if calibration is not None else None
            if anything_estimated and score not in ("gcv", "evidence", "sure"):
                raise ValueError(
                    "weight: auto requires solver.calibration.score in {gcv, evidence, sure} "
                    "when imaging-system parameters are estimated"
                )

        if self.object.prior == "none" and self.reconstruction_kind == "closed_form":
            raise ValueError("object.prior: none with solver.reconstruction: closed_form is an unregularized inverse")
        if (
            self.object.prior == "tv" or isinstance(self.object.prior, PriorDenoiser) or self.object.constraints
        ) and self.reconstruction_kind == "closed_form":
            raise ValueError(
                "object.prior tv/denoiser and constraints: [nonneg] have no closed form; "
                "choose fista, admm, or richardson_lucy under solver.reconstruction"
            )
        if self.reconstruction_kind == "richardson_lucy":
            if self.noise.kind != "poisson":
                raise ValueError("richardson_lucy requires noise: poisson")
            if self.object.prior != "none":
                raise ValueError("richardson_lucy requires object.prior: none")
        return self

    @property
    def reconstruction_kind(self) -> str:
        """Bare name of the chosen reconstruction option."""
        recon = self.solver.reconstruction
        if isinstance(recon, str):
            return recon
        return next(iter(recon.__class__.model_fields))

    @property
    def contrast(self) -> str:
        return self.imaging_system.contrast


def free_parameters(system: ImagingSystem) -> dict[str, FreeParameter]:
    """Collect every :class:`FreeParameter` of an imaging system.

    Parameters
    ----------
    system : ImagingSystem
        The ``imaging_system`` block.

    Returns
    -------
    dict[str, FreeParameter]
        Mapping from dotted parameter name (e.g.
        ``"detection.numerical_aperture"``,
        ``"detection.pupil_aberrations.spherical"``) to its prior. The
        ``zernike_up_to_order`` shorthand is expanded into named terms by
        :mod:`waveorder.calibrate.zernike_names` consumers, not here; it
        appears under its own key when present with a tolerance.
    """
    found: dict[str, FreeParameter] = {}

    def visit(obj: Any, prefix: str) -> None:
        if isinstance(obj, FreeParameter):
            found[prefix] = obj
            return
        if isinstance(obj, PupilAberrations):
            shorthand = obj.zernike_up_to_order
            if shorthand is not None:
                found[f"{prefix}.zernike_up_to_order"] = FreeParameter(
                    nominal=0.0, tolerance=shorthand.tolerance, bounds=shorthand.bounds
                )
            for name, value in obj.terms.items():
                visit(value, f"{prefix}.{name}")
            return
        if isinstance(obj, BaseModel):
            for name in obj.__class__.model_fields:
                value = getattr(obj, name)
                if isinstance(value, (BaseModel,)):
                    visit(value, f"{prefix}.{name}" if prefix else name)
                elif isinstance(value, FreeParameter):
                    visit(value, f"{prefix}.{name}" if prefix else name)

    visit(system, "")
    return found
