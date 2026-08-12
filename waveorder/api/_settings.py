"""Shared base settings classes used across reconstruction types."""

from __future__ import annotations

import warnings
from typing import Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_serializer,
    field_validator,
    model_validator,
)

from waveorder._pixel_size import YXPixelSize
from waveorder.optim._types import OptimizableFloat


def _float_val(v) -> float:
    """Extract float value from float or OptimizableFloat."""
    if isinstance(v, OptimizableFloat):
        return v.value
    return float(v)


class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WavelengthIllumination(MyBaseModel):
    wavelength_illumination: PositiveFloat = Field(default=0.532, description="illumination wavelength in micrometers")


class FourierTransferFunctionSettings(MyBaseModel):
    yx_pixel_size: YXPixelSize = Field(
        default_factory=lambda: YXPixelSize.isotropic(0.1),
        description="lateral pixel size in micrometers; scalar for isotropic, or {y, x} mapping for anisotropic",
    )
    z_pixel_size: PositiveFloat = Field(default=0.25, description="axial pixel size in micrometers")
    z_padding: NonNegativeInt = Field(default=0, description="z slices to pad for axial boundary effects")
    z_focus_offset: float = Field(
        default=0,
        description="offset from center slice in slice units",
    )
    index_of_refraction_media: PositiveFloat = Field(default=1.3, description="refractive index of imaging media")
    numerical_aperture_detection: PositiveFloat = Field(
        default=1.2, description="detection objective numerical aperture"
    )

    @field_validator("yx_pixel_size", mode="before")
    @classmethod
    def _normalize_yx_pixel_size(cls, v):
        return YXPixelSize.from_value(v)

    @field_serializer("yx_pixel_size")
    def _serialize_yx_pixel_size(self, value):
        # Robust to model_copy(update={...}) which can leave a scalar in
        # place of a YXPixelSize; normalize before dumping.
        return YXPixelSize.from_value(value).model_dump()

    @model_validator(mode="after")
    def validate_numerical_aperture_detection(self):
        na_det = _float_val(self.numerical_aperture_detection)
        if na_det > self.index_of_refraction_media:
            raise ValueError(
                f"numerical_aperture_detection = {na_det} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
            )
        return self

    @model_validator(mode="after")
    def warn_pixel_size_consistency(self):
        # Normalize defensively so model_copy(update={"yx_pixel_size": 0.2}),
        # which bypasses field validators, still produces a usable value.
        yx = YXPixelSize.from_value(self.yx_pixel_size)
        for axis, ps in (("y", yx.y), ("x", yx.x)):
            ratio = ps / self.z_pixel_size
            if ratio < 1.0 / 20 or ratio > 20:
                warnings.warn(
                    f"{axis}_pixel_size ({ps}) / z_pixel_size ({self.z_pixel_size}) = {ratio}. Did you use consistent units?",
                    UserWarning,
                )
        return self


class OptimizableFourierTransferFunctionSettings(FourierTransferFunctionSettings):
    """FourierTransferFunctionSettings with OptimizableFloat support and tilt fields.

    Used by phase and fluorescence settings (not birefringence).
    """

    z_focus_offset: Union[float, OptimizableFloat] = Field(
        default=0,
        description="(optimizable) offset from center slice in slice units",
    )
    numerical_aperture_detection: Union[PositiveFloat, OptimizableFloat] = Field(
        default=1.2, description="(optimizable) detection objective numerical aperture"
    )
    tilt_angle_zenith: Union[float, OptimizableFloat] = Field(
        default=0.0, description="(optimizable) illumination tilt zenith angle in radians"
    )
    tilt_angle_azimuth: Union[float, OptimizableFloat] = Field(
        default=0.0, description="(optimizable) illumination tilt azimuth angle in radians"
    )

    @model_validator(mode="after")
    def validate_positive_na_detection(self):
        val = _float_val(self.numerical_aperture_detection)
        if val <= 0:
            raise ValueError(f"numerical_aperture_detection must be positive, got {val}")
        return self

    def resolve_floats(self):
        """Return a copy with OptimizableFloat fields resolved to plain floats."""
        d = {}
        for name in self.__class__.model_fields:
            v = getattr(self, name)
            d[name] = v.value if isinstance(v, OptimizableFloat) else v
        return self.__class__.model_validate(d)


class FourierApplyInverseSettings(MyBaseModel):
    # "RL"/"RLGC" are accepted here but only implemented for 3D fluorescence
    # (see waveorder.api.fluorescence); other modalities raise NotImplementedError.
    reconstruction_algorithm: Literal["Tikhonov", "TV", "RL", "RLGC"] = Field(
        default="Tikhonov",
        description="'Tikhonov'/'TV' regularization, or 'RL'/'RLGC' iterative deconvolution "
        "(3D fluorescence only)",
    )
    regularization_strength: NonNegativeFloat = Field(default=1e-3, description="strength of regularization")
    TV_rho_strength: PositiveFloat = Field(default=1e-3, description="ADMM rho parameter for TV regularization")
    TV_iterations: NonNegativeInt = Field(default=1, description="ADMM iterations for TV regularization")

    def to_model_kwargs(self) -> dict:
        """Flatten to the keyword arguments of ``apply_inverse_transfer_function``.

        The config groups related knobs into blocks so a YAML only carries the
        ones its algorithm reads; the model functions take one flat signature.
        This is the seam between the two.
        """
        return self.model_dump()
