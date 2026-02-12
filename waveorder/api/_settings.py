"""Shared base settings classes used across reconstruction types."""

import warnings
from typing import Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)


class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WavelengthIllumination(MyBaseModel):
    wavelength_illumination: PositiveFloat = 0.532


class FourierTransferFunctionSettings(MyBaseModel):
    yx_pixel_size: PositiveFloat = 6.5 / 20
    z_pixel_size: PositiveFloat = 2.0
    z_padding: NonNegativeInt = 0
    z_focus_offset: Union[float, Literal["auto"]] = 0
    index_of_refraction_media: PositiveFloat = 1.3
    numerical_aperture_detection: PositiveFloat = 1.2

    @model_validator(mode="after")
    def validate_numerical_aperture_detection(self):
        if self.numerical_aperture_detection > self.index_of_refraction_media:
            raise ValueError(
                f"numerical_aperture_detection = {self.numerical_aperture_detection} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
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


class FourierApplyInverseSettings(MyBaseModel):
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov"
    regularization_strength: NonNegativeFloat = 1e-3
    TV_rho_strength: PositiveFloat = 1e-3
    TV_iterations: NonNegativeInt = 1
