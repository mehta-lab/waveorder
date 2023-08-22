import os
from typing import List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    root_validator,
    validator,
)

# This file defines the configuration settings for the CLI.

# Example settings files in `/examples/settings/` are autmatically generated
# by the tests in `/tests/cli_tests/test_settings.py` - `test_generate_example_settings`.

# To keep the example settings up to date, run `pytest` locally when this file changes.


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel, extra=Extra.forbid):
    pass


# Bottom level settings
class WavelengthIllumination(MyBaseModel):
    wavelength_illumination: PositiveFloat = 0.532


class BirefringenceTransferFunctionSettings(MyBaseModel):
    swing: float = 0.1

    @validator("swing")
    def swing_range(cls, v):
        if v <= 0 or v >= 1.0:
            raise ValueError(f"swing = {v} should be between 0 and 1.")
        return v


class BirefringenceApplyInverseSettings(WavelengthIllumination):
    background_path: str = ""
    remove_estimated_background: bool = False
    flip_orientation: bool = False
    rotate_orientation: bool = False

    @validator("background_path")
    def check_background_path(cls, v):
        if v == "":
            return v

        raw_dir = r"{}".format(v)
        if not os.path.isdir(raw_dir):
            raise ValueError(f"{v} is not a existing directory")
        return raw_dir


class FourierTransferFunctionSettings(MyBaseModel):
    yx_pixel_size: PositiveFloat = 6.5 / 20
    z_pixel_size: PositiveFloat = 2.0
    z_padding: NonNegativeInt = 0
    index_of_refraction_media: PositiveFloat = 1.3
    numerical_aperture_detection: PositiveFloat = 1.2

    @validator("numerical_aperture_detection")
    def na_det(cls, v, values):
        n = values["index_of_refraction_media"]
        if v > n:
            raise ValueError(
                f"numerical_aperture_detection = {v} must be less than or equal to index_of_refraction_media = {n}"
            )
        return v

    @validator("z_pixel_size")
    def warn_unit_consistency(cls, v, values):
        yx_pixel_size = values["yx_pixel_size"]
        ratio = yx_pixel_size / v
        if ratio < 1.0 / 20 or ratio > 20:
            raise Warning(
                f"yx_pixel_size ({yx_pixel_size}) / z_pixel_size ({v}) = {ratio}. Did you use consistent units?"
            )
        return v


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

    @validator("numerical_aperture_illumination")
    def na_ill(cls, v, values):
        n = values.get("index_of_refraction_media")
        if v > n:
            raise ValueError(
                f"numerical_aperture_illumination = {v} must be less than or equal to index_of_refraction_media = {n}"
            )
        return v


class FluorescenceTransferFunctionSettings(FourierTransferFunctionSettings):
    wavelength_emission: PositiveFloat = 0.507

    @validator("wavelength_emission")
    def warn_unit_consistency(cls, v, values):
        yx_pixel_size = values.get("yx_pixel_size")
        ratio = yx_pixel_size / v
        if ratio < 1.0 / 20 or ratio > 20:
            raise Warning(
                f"yx_pixel_size ({yx_pixel_size}) / wavelength_illumination ({v}) = {ratio}. Did you use consistent units?"
            )
        return v


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
    birefringence: Optional[BirefringenceSettings]
    phase: Optional[PhaseSettings]
    fluorescence: Optional[FluorescenceSettings]

    @root_validator(pre=False)
    def validate_reconstruction_types(cls, values):
        if (values.get("birefringence") or values.get("phase")) and values.get(
            "fluorescence"
        ) is not None:
            raise ValueError(
                '"fluorescence" cannot be present alongside "birefringence" or "phase". Please use one configuration file for a "fluorescence" reconstruction and another configuration file for a "birefringence" and/or "phase" reconstructions.'
            )
        num_channel_names = len(values.get("input_channel_names"))
        if values.get("birefringence") is None:
            if (
                values.get("phase") is None
                and values.get("fluorescence") is None
            ):
                raise ValueError(
                    "Provide settings for either birefringence, phase, birefringence + phase, or fluorescence."
                )
            if num_channel_names != 1:
                raise ValueError(
                    f"{num_channel_names} channels names provided. Please provide a single channel for fluorescence/phase reconstructions."
                )

        return values
