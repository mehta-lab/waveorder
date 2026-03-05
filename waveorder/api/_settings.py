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
    model_validator,
)

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
    yx_pixel_size: PositiveFloat = Field(default=0.1, description="lateral pixel size in micrometers")
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

    @model_validator(mode="after")
    def validate_numerical_aperture_detection(self):
        na_det = _float_val(self.numerical_aperture_detection)
        if na_det > self.index_of_refraction_media:
            raise ValueError(
                f"numerical_aperture_detection = {na_det} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
            )
        return self

    @model_validator(mode="after")
    def warn_unit_consistency(self):
        ratio = self.yx_pixel_size / self.z_pixel_size
        if ratio < 1.0 / 20 or ratio > 20:
            warnings.warn(
                f"yx_pixel_size ({self.yx_pixel_size}) / z_pixel_size ({self.z_pixel_size}) = {ratio}. Did you use consistent units?",
                UserWarning,
            )
        return self


class OptimizableFourierTransferFunctionSettings(FourierTransferFunctionSettings):
    """FourierTransferFunctionSettings with OptimizableFloat support and tilt fields.

    Used by phase and fluorescence settings (not birefringence).
    """

    z_focus_offset: Union[float, OptimizableFloat] = Field(
        default=0,
        description="[o] offset from center slice in slice units",
    )
    numerical_aperture_detection: Union[PositiveFloat, OptimizableFloat] = Field(
        default=1.2, description="[o] detection objective numerical aperture"
    )
    tilt_angle_zenith: Union[float, OptimizableFloat] = Field(
        default=0.0, description="[o] illumination tilt zenith angle in radians"
    )
    tilt_angle_azimuth: Union[float, OptimizableFloat] = Field(
        default=0.0, description="[o] illumination tilt azimuth angle in radians"
    )


class FourierApplyInverseSettings(MyBaseModel):
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = Field(
        default="Tikhonov",
        description="'Tikhonov' or 'TV' regularization",
    )
    regularization_strength: NonNegativeFloat = Field(default=1e-3, description="strength of regularization")
    TV_rho_strength: PositiveFloat = Field(default=1e-3, description="ADMM rho parameter for TV regularization")
    TV_iterations: NonNegativeInt = Field(default=1, description="ADMM iterations for TV regularization")
