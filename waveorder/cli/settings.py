from typing import List, Literal, Optional, Union

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from waveorder.api._settings import (  # noqa: F401
    FourierApplyInverseSettings,
    FourierTransferFunctionSettings,
    MyBaseModel,
)
from waveorder.api.birefringence import (  # noqa: F401
    ApplyInverseSettings as BirefringenceApplyInverseSettings,
)
from waveorder.api.birefringence import Settings as BirefringenceSettings
from waveorder.api.fluorescence import (  # noqa: F401
    Settings as FluorescenceSettings,
)
from waveorder.api.phase import Settings as PhaseSettings  # noqa: F401


class MidbandPowerLoss(MyBaseModel):
    type: Literal["midband_power"] = "midband_power"
    midband_fractions: tuple[float, float] = Field(
        default=(0.125, 0.25), description="inner/outer fractions of cutoff frequency"
    )


class OptimizationSettings(MyBaseModel):
    num_iterations: PositiveInt = Field(default=10, description="number of Adam optimizer steps")
    loss: MidbandPowerLoss = Field(default_factory=MidbandPowerLoss, description="loss function configuration")
    log_dir: Optional[str] = Field(default=None, description="TensorBoard log directory (null = no logging)")



# Top level settings (CLI-specific)
class ReconstructionSettings(MyBaseModel):
    input_channel_names: List[str] = Field(
        default=[f"State{i}" for i in range(4)],
        description="names of input channels in the dataset",
    )
    time_indices: Union[NonNegativeInt, List[NonNegativeInt], Literal["all"]] = Field(
        default="all", description="time points to reconstruct"
    )
    reconstruction_dimension: Literal[2, 3] = Field(default=3, description="2 for thin samples, 3 for thick")
    birefringence: Optional[BirefringenceSettings] = None
    phase: Optional[PhaseSettings] = None
    fluorescence: Optional[FluorescenceSettings] = None
    optimization: Optional[OptimizationSettings] = None

    @model_validator(mode="after")
    def validate_reconstruction_types(self):
        if (self.birefringence or self.phase) and self.fluorescence is not None:
            raise ValueError(
                '"fluorescence" cannot be present alongside "birefringence" or "phase". Please use one configuration file for a "fluorescence" reconstruction and another configuration file for a "birefringence" and/or "phase" reconstructions.'
            )
        num_channel_names = len(self.input_channel_names)
        if self.birefringence is None:
            if self.phase is None and self.fluorescence is None:
                raise ValueError(
                    "Provide settings for either birefringence, phase, birefringence + phase, or fluorescence."
                )
            if num_channel_names != 1:
                raise ValueError(
                    f"{num_channel_names} channels names provided. Please provide a single channel for fluorescence/phase reconstructions."
                )

        return self

    @property
    def output_channel_names(self) -> list[str]:
        from waveorder.api._utils import _output_channel_names

        return _output_channel_names(
            recon_biref=self.birefringence is not None,
            recon_phase=self.phase is not None,
            recon_fluo=self.fluorescence is not None,
            recon_dim=self.reconstruction_dimension,
            fluor_channel_name=self.input_channel_names[0],
        )

    @property
    def output_z_is_singleton(self) -> bool:
        return self.reconstruction_dimension == 2
