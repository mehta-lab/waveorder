import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)

# This file defines the configuration settings for the CLI.

# Example settings files in `/docs/examples/settings/` are autmatically generated
# by the tests in `/tests/cli_tests/test_settings.py` - `test_generate_example_settings`.

# To keep the example settings up to date, run `pytest` locally when this file changes.


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Bottom level settings
class WavelengthIllumination(MyBaseModel):
    wavelength_illumination: PositiveFloat = 0.532


class BirefringenceTransferFunctionSettings(MyBaseModel):
    swing: float = 0.1

    @field_validator("swing")
    @classmethod
    def swing_range(cls, v):
        if v <= 0 or v >= 1.0:
            raise ValueError(f"swing = {v} should be between 0 and 1.")
        return v


class BirefringenceApplyInverseSettings(WavelengthIllumination):
    background_path: Union[str, Path] = ""

    @field_validator("background_path")
    @classmethod
    def check_background_path(cls, v):
        if v == "":
            return v

        raw_dir = r"{}".format(v)
        if not os.path.isdir(raw_dir):
            raise ValueError(f"{v} is not a existing directory")
        return raw_dir

    remove_estimated_background: bool = False
    flip_orientation: bool = False
    rotate_orientation: bool = False


class FourierTransferFunctionSettings(MyBaseModel):
    yx_pixel_size: PositiveFloat = 6.5 / 20
    z_pixel_size: PositiveFloat = 2.0
    z_padding: NonNegativeInt = 0
    z_focus_offset: Union[int, Literal["auto"]] = 0
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


class PhaseTransferFunctionSettings(
    FourierTransferFunctionSettings,
    WavelengthIllumination,
):
    numerical_aperture_illumination: NonNegativeFloat = 0.5
    invert_phase_contrast: bool = False
    illumination_sector_angles: Optional[List[Tuple[float, float]]] = None

    @model_validator(mode="after")
    def validate_numerical_aperture_illumination(self):
        if (
            self.numerical_aperture_illumination
            > self.index_of_refraction_media
        ):
            raise ValueError(
                f"numerical_aperture_illumination = {self.numerical_aperture_illumination} must be less than or equal to index_of_refraction_media = {self.index_of_refraction_media}"
            )
        return self

    @field_validator("illumination_sector_angles")
    @classmethod
    def validate_sector_angles(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError(
                "illumination_sector_angles must contain at least one sector"
            )
        normalized = []
        for start, end in v:
            if start >= end:
                raise ValueError(
                    f"Sector start angle {start} must be less than end angle {end}"
                )
            # Normalize angles to [0, 360) using modulo 360
            # Special case: preserve 360 for full aperture (don't reduce to 0)
            normalized_start = start % 360
            normalized_end = end % 360 if end % 360 != 0 else 360
            normalized.append((normalized_start, normalized_end))
        return normalized


class FluorescenceTransferFunctionSettings(FourierTransferFunctionSettings):
    wavelength_emission: PositiveFloat = 0.507
    confocal_pinhole_diameter: Optional[PositiveFloat] = None

    @model_validator(mode="after")
    def warn_unit_consistency(self):
        ratio = self.yx_pixel_size / self.wavelength_emission
        if ratio < 1.0 / 20 or ratio > 20:
            warnings.warn(
                f"yx_pixel_size ({self.yx_pixel_size}) / wavelength_illumination ({self.wavelength_emission}) = {ratio}. Did you use consistent units?",
                UserWarning,
            )
        return self


# Second level settings
class BirefringenceSettings(MyBaseModel):
    transfer_function: BirefringenceTransferFunctionSettings = (
        BirefringenceTransferFunctionSettings()
    )
    apply_inverse: BirefringenceApplyInverseSettings = (
        BirefringenceApplyInverseSettings()
    )


class PhaseSettings(MyBaseModel):
    transfer_function: PhaseTransferFunctionSettings = (
        PhaseTransferFunctionSettings()
    )
    apply_inverse: FourierApplyInverseSettings = FourierApplyInverseSettings()


class FluorescenceSettings(MyBaseModel):
    transfer_function: FluorescenceTransferFunctionSettings = (
        FluorescenceTransferFunctionSettings()
    )
    apply_inverse: FourierApplyInverseSettings = FourierApplyInverseSettings()


# Top level settings
class ReconstructionSettings(MyBaseModel):
    input_channel_names: List[str] = [f"State{i}" for i in range(4)]
    time_indices: Union[
        NonNegativeInt, List[NonNegativeInt], Literal["all"]
    ] = "all"
    reconstruction_dimension: Literal[2, 3] = 3
    birefringence: Optional[BirefringenceSettings] = None
    phase: Optional[PhaseSettings] = None
    fluorescence: Optional[FluorescenceSettings] = None

    @model_validator(mode="after")
    def validate_reconstruction_types(self):
        if (
            self.birefringence or self.phase
        ) and self.fluorescence is not None:
            raise ValueError(
                '"fluorescence" cannot be present alongside "birefringence" or "phase". Please use one configuration file for a "fluorescence" reconstruction and another configuration file for a "birefringence" and/or "phase" reconstructions.'
            )
        num_channel_names = len(self.input_channel_names)

        # Check for sector illumination in phase reconstruction
        phase_settings = self.phase
        if phase_settings is not None:
            sector_angles = (
                phase_settings.transfer_function.illumination_sector_angles
            )
            if sector_angles is not None:
                # Multi-channel reconstruction with sector illumination
                if len(sector_angles) != num_channel_names:
                    raise ValueError(
                        f"Number of illumination_sector_angles ({len(sector_angles)}) must match number of input_channel_names ({num_channel_names})"
                    )
            else:
                # Single channel phase reconstruction without sector illumination
                if self.birefringence is None and num_channel_names != 1:
                    raise ValueError(
                        f"{num_channel_names} channels names provided. Please provide a single channel for phase reconstructions without sector illumination."
                    )
        else:
            if self.birefringence is None:
                if self.fluorescence is None:
                    raise ValueError(
                        "Provide settings for either birefringence, phase, birefringence + phase, or fluorescence."
                    )
                if num_channel_names != 1:
                    raise ValueError(
                        f"{num_channel_names} channels names provided. Please provide a single channel for fluorescence reconstructions."
                    )

        return self
