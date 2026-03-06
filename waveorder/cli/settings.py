from typing import Any, List, Literal, Optional, Union

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
from waveorder.optim.losses import MidbandPowerLossSettings, _LossBaseModel

_LOSS_TYPE_MAP = {
    "midband_power": "MidbandPowerLossSettings",
    "total_variation": "TotalVariationLossSettings",
    "laplacian_variance": "LaplacianVarianceLossSettings",
    "normalized_variance": "NormalizedVarianceLossSettings",
    "spectral_flatness": "SpectralFlatnessLossSettings",
}


def _parse_loss(v: Any):
    """Parse a loss config dict or instance into a LossSettings object."""
    if isinstance(v, dict):
        from waveorder.optim import losses

        loss_type = v.get("type", "midband_power")
        cls_name = _LOSS_TYPE_MAP.get(loss_type)
        if cls_name is None:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return getattr(losses, cls_name)(**v)
    return v


class OptimizationSettings(MyBaseModel):
    max_iterations: PositiveInt = Field(default=10, description="maximum optimizer steps (ignored by grid_search)")
    method: str = Field(default="adam", description="optimizer method: adam, lbfgs, nelder_mead, grid_search")
    convergence_tol: Optional[float] = Field(default=None, description="early stopping tolerance")
    convergence_patience: Optional[PositiveInt] = Field(default=5, description="patience for early stopping")
    use_gradients: Optional[bool] = Field(default=None, description="auto-detect from method if null")
    grid_points: int = Field(default=7, description="grid points per parameter (grid_search only)")
    loss: _LossBaseModel = Field(default_factory=MidbandPowerLossSettings, description="loss function configuration")
    log_dir: Optional[str] = Field(default=None, description="TensorBoard log directory (null = no logging)")

    @model_validator(mode="before")
    @classmethod
    def _parse_loss_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "loss" in data:
            data["loss"] = _parse_loss(data["loss"])
        return data


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
