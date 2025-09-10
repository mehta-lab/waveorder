from datetime import datetime

import napari
import numpy as np
import torch

# Commenting biahub dependency for now
# from biahub.cli.utils import model_to_yaml
# from biahub.settings import StitchSettings
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from torch.utils.tensorboard import SummaryWriter

from waveorder import optics, util
from waveorder.models import isotropic_thin_3d


# === Core Functions ===
def run_reconstruction(
    zyx_tile: torch.Tensor, recon_args: dict
) -> torch.Tensor:

    # Prepare transfer function arguments
    tf_args = recon_args.copy()
    Z, _, _ = zyx_tile.shape
    tf_args["z_position_list"] = (
        torch.arange(Z) - (Z // 2) + recon_args["z_offset"]
    ) * recon_args["z_scale"]
    tf_args.pop("z_offset")
    tf_args.pop("z_scale")

    # Core reconstruction calls
    tf_abs, tf_phase = isotropic_thin_3d.calculate_transfer_function(**tf_args)
    system = isotropic_thin_3d.calculate_singular_system(tf_abs, tf_phase)
    _, yx_phase_recon = isotropic_thin_3d.apply_inverse_transfer_function(
        zyx_tile, system, regularization_strength=1e-2
    )
    return yx_phase_recon


def compute_midband_power(
    yx_array: torch.Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    band: tuple[float, float] = (0.125, 0.25),
) -> torch.Tensor:
    _, _, fxx, fyy = util.gen_coordinate(yx_array.shape, pixel_size)
    frr = torch.tensor(np.sqrt(fxx**2 + fyy**2))
    xy_abs_fft = torch.abs(torch.fft.fftn(yx_array))
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(frr > cutoff * band[0], frr < cutoff * band[1])
    return torch.sum(xy_abs_fft[mask])


def extract_tiles(
    zyx_data: np.ndarray, num_tiles: tuple[int, int], overlap_pct: float
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int, int]]]:
    Z, Y, X = zyx_data.shape
    tile_height = int(
        np.ceil(Y / (num_tiles[0] - (num_tiles[0] - 1) * overlap_pct))
    )
    tile_width = int(
        np.ceil(X / (num_tiles[1] - (num_tiles[1] - 1) * overlap_pct))
    )
    stride_y = int(tile_height * (1 - overlap_pct))
    stride_x = int(tile_width * (1 - overlap_pct))

    tiles = {}
    translations = {}
    for yi in range(num_tiles[0]):
        for xi in range(num_tiles[1]):
            y0, x0 = yi * stride_y, xi * stride_x
            y1, x1 = min(y0 + tile_height, Y), min(x0 + tile_width, X)
            tile_name = f"0/0/{yi:03d}{xi:03d}"
            tiles[tile_name] = zyx_data[:, y0:y1, x0:x1]
            translations[tile_name] = (0, y0, x0)
    return tiles, translations


def log_optimization_progress(
    step: int,
    optimization_params: dict[str, torch.nn.Parameter],
    loss: torch.Tensor,
    tb_writer: SummaryWriter,
    recon_args: dict,
    yx_recon: torch.Tensor,
) -> None:
    # Print progress
    print(f"Step {step + 1}/{NUM_ITERATIONS}")
    for name, param in optimization_params.items():
        print(f"\t{name} = {param.item():.4f}")
    print(f"\tLoss: {loss.item():.2e}\n")

    # Log metrics and images
    tb_writer.add_scalar("Loss", loss.item(), step)
    for name, param in optimization_params.items():
        tb_writer.add_scalar(name, param.item(), step)

    yx_pixel_factor = 2
    fyy, fxx = util.generate_frequencies(
        [yx_pixel_factor * x for x in recon_args["yx_shape"]],
        recon_args["yx_pixel_size"] / yx_pixel_factor,
    )
    pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        recon_args["numerical_aperture_illumination"],
        recon_args["wavelength_illumination"],
        recon_args["index_of_refraction_media"],
        recon_args["tilt_angle_zenith"],
        recon_args["tilt_angle_azimuth"],
    )
    tb_writer.add_image(
        "Illumination Pupil",
        torch.fft.fftshift(pupil).detach().numpy()[None],
        step,
    )
    tb_writer.add_image(
        "Reconstructed Phase", yx_recon.detach().numpy()[None], step
    )


def prepare_optimizer(
    optimizable_params: dict[str, tuple[bool, float, float]],
) -> tuple[dict[str, torch.nn.Parameter], torch.optim.Optimizer]:
    optimization_params: dict[str, torch.nn.Parameter] = {}
    optimizer_config = []
    for name, (enabled, initial, lr) in optimizable_params.items():
        if enabled:
            param = torch.nn.Parameter(
                torch.tensor([initial], device="cpu"), requires_grad=True
            )
            optimization_params[name] = param
            optimizer_config.append({"params": [param], "lr": lr})

    optimizer = torch.optim.Adam(optimizer_config)
    return optimization_params, optimizer


def optimize_tile(
    zyx_tile: torch.Tensor,
    recon_args: dict,
    optimizable_params: dict[str, tuple[bool, float, float]],
    tb_writer: SummaryWriter,
    num_iterations: int = 10,
) -> torch.Tensor:
    optimization_params, optimizer = prepare_optimizer(optimizable_params)

    for step in range(num_iterations):

        # Update params
        for name, param in optimization_params.items():
            recon_args[name] = param

        # Run reconstruction and compute loss
        yx_recon = run_reconstruction(zyx_tile, recon_args)
        loss = -compute_midband_power(
            yx_recon,
            NA_det=0.15,
            lambda_ill=recon_args["wavelength_illumination"],
            pixel_size=recon_args["yx_pixel_size"],
            band=(0.1, 0.2),
        )

        # Update optimizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_optimization_progress(
            step, optimization_params, loss, tb_writer, recon_args, yx_recon
        )

    return yx_recon.detach()


# === Configuration ===
# INPUTS
INPUT_PATH = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/0-convert/live_imaging/tracking_symlink.zarr"
INPUT_FOV = "A/1/001007"
SUBTILES = ["0/0/001001"]  # or "all"

# OUTPUTS
OUTPUT_PATH = "./optimized_recon.zarr"
OUTPUT_CHANNEL_NAME = "recon"

# TILING
STITCH_CONFIG_PATH = "./stitch_config.yaml"
NUM_TILES = (6, 6)
OVERLAP_FRACTION = 0.2

# OPTIMIZATION
NUM_ITERATIONS = 10
LOGS_DIR = "./runs"
FIXED_PARAMS = {
    "wavelength_illumination": 0.450,
    "index_of_refraction_media": 1.0,
    "invert_phase_contrast": True,
}
OPTIMIZABLE_PARAMS = {  # (optimize?, initial_value, learning_rate)
    "z_offset": (True, 0.0, 0.01),
    "numerical_aperture_detection": (True, 0.15, 0.001),
    "numerical_aperture_illumination": (True, 0.1, 0.001),
    "tilt_angle_zenith": (True, 0.1, 0.005),
    "tilt_angle_azimuth": (True, 260 * np.pi / 180, 0.001),
}

# === Main Execution ===
input_store = open_ome_zarr(INPUT_PATH)
zyx_data = input_store[INPUT_FOV].data[0][0]
_, _, z_scale, y_scale, x_scale = input_store[INPUT_FOV].scale

output_store = open_ome_zarr(
    OUTPUT_PATH, layout="hcs", mode="w", channel_names=[OUTPUT_CHANNEL_NAME]
)
tiles, translations = extract_tiles(zyx_data, NUM_TILES, OVERLAP_FRACTION)
# Commenting biahub dependency for now
# model_to_yaml(
#     StitchSettings(total_translation=translations), STITCH_CONFIG_PATH
# )

if SUBTILES == "all":
    selected_keys = tiles.keys()
else:
    selected_keys = SUBTILES

for key in selected_keys:
    zyx_tile = torch.tensor(tiles[key], dtype=torch.float32, device="cpu")

    print(f"Processing tile {key}")
    timestamp = datetime.now().strftime("%d%H%M")
    log_dir = f"{LOGS_DIR}/tile_{key.replace('/', '_')}_{timestamp}"
    tb_writer = SummaryWriter(log_dir=log_dir)

    # Prepare reconstruction arguments
    recon_args = FIXED_PARAMS
    for name, value in OPTIMIZABLE_PARAMS.items():
        recon_args[name] = torch.tensor(
            [value[1]], dtype=torch.float32, device="cpu"
        )
    recon_args["yx_shape"] = zyx_tile.shape[1:]
    recon_args["yx_pixel_size"] = y_scale
    recon_args["z_scale"] = z_scale

    initial_recon = run_reconstruction(zyx_tile, recon_args)
    optimized_recon = optimize_tile(
        zyx_tile,
        recon_args,
        OPTIMIZABLE_PARAMS,
        tb_writer,
        num_iterations=NUM_ITERATIONS,
    )
    tb_writer.close()

    # Write to napari viewer
    scale = [z_scale, y_scale, x_scale]
    viewer = napari.Viewer()
    viewer.add_image(
        initial_recon.numpy()[None], name=f"initial-{key}", scale=scale
    )
    viewer.add_image(
        optimized_recon.numpy()[None], name=f"optimized-{key}", scale=scale
    )
    viewer.add_image(zyx_tile, name=f"tile-{key}", scale=scale)

    # Write to output store
    pos = output_store.create_position(*key.split("/"))
    pos.create_image(
        "0",
        optimized_recon[None, None, None].numpy(),
        transform=[TransformationMeta(type="scale", scale=[1, 1] + scale)],
    )
    input("Press Enter to continue...")
