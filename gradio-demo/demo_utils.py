"""
Utility functions for Gradio demos

Provides reusable components for:
- Data loading from OME-Zarr stores
- Image normalization and processing
- Slice extraction from xarray DataArrays
- Phase reconstruction and optimization

Design Notes
------------
All image processing functions work with xarray.DataArray to maintain
labeled dimensions and coordinate information as long as possible.
Only convert to numpy arrays at the final display step.
"""

from pathlib import Path
from typing import Generator

import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray
from xarray_ome import open_ome_dataset

from waveorder import util
from waveorder.models import isotropic_thin_3d

# Type alias for device specification
Device = torch.device | str | None


def get_device(device: Device = None) -> torch.device:
    """
    Get torch device with smart defaults.

    Parameters
    ----------
    device : torch.device | str | None
        If None, auto-selects cuda if available, else cpu.
        If str, converts to torch.device.
        If torch.device, returns as-is.

    Returns
    -------
    torch.device
        Validated device ready for use

    Examples
    --------
    >>> get_device()  # Auto-detect
    device(type='cuda', index=0)  # if GPU available

    >>> get_device("cpu")  # Force CPU
    device(type='cpu')

    >>> get_device(torch.device("cuda:1"))  # Specific GPU
    device(type='cuda', index=1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(device)}")
            gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"   GPU Memory: {gpu_mem_gb:.2f} GB")
        else:
            print("💻 Using CPU (GPU not available)")
        return device

    if isinstance(device, str):
        return torch.device(device)

    return device


# === HCS Plate Loading ===
def get_plate_metadata(zarr_path: Path | str) -> dict:
    """
    Extract HCS plate metadata for FOV selection.

    Parameters
    ----------
    zarr_path : Path | str
        Path to the HCS plate zarr store

    Returns
    -------
    dict
        Metadata with keys:
        - 'rows': list of row names (e.g., ['A', 'B'])
        - 'columns': list of column names (e.g., ['1', '2', '3'])
        - 'wells': dict mapping (row, col) to list of field names
        - 'plate': ngff_zarr plate object for later access
    """
    import ngff_zarr as nz

    plate = nz.from_hcs_zarr(str(zarr_path))

    # Extract row and column names
    rows = [r.name for r in plate.metadata.rows]
    columns = [c.name for c in plate.metadata.columns]

    # Build wells dictionary
    wells = {}
    for well_meta in plate.metadata.wells:
        row_name = plate.metadata.rows[well_meta.rowIndex].name
        col_name = plate.metadata.columns[well_meta.columnIndex].name

        well = plate.get_well(row_name, col_name)
        fields = [img.path for img in well.images]

        wells[(row_name, col_name)] = fields

    return {
        "rows": rows,
        "columns": columns,
        "wells": wells,
        "plate": plate,
    }


def load_fov_from_plate(
    plate, row: str, column: str, field: str, resolution: int = 0
) -> xr.DataArray:
    """
    Load a specific FOV from HCS plate.

    Parameters
    ----------
    plate : ngff_zarr plate object
        Plate loaded with from_hcs_zarr()
    row : str
        Row name (e.g., 'A')
    column : str
        Column name (e.g., '1')
    field : str
        Field/position name (e.g., '001007')
    resolution : int, optional
        Resolution level to load, by default 0

    Returns
    -------
    xr.DataArray
        Image data with labeled dimensions (T, C, Z, Y, X)
    """
    well = plate.get_well(row, column)

    # Find the image with matching field name
    field_index = None
    for i, img_meta in enumerate(well.images):
        if img_meta.path == field:
            field_index = i
            break

    if field_index is None:
        raise ValueError(f"Field '{field}' not found in well {row}/{column}")

    # Get the image
    image = well.get_image(field_index)

    # Convert to xarray (assuming single channel for now)
    # Shape is (T, C, Z, Y, X)
    data_array = image.images[0].data

    # Create xarray with proper coordinates
    T, C, Z, Y, X = data_array.shape

    # Get scales from metadata
    # ngff_zarr uses different structure - access from image metadata
    coords = {
        "T": np.arange(T),
        "C": np.arange(C),
        "Z": np.arange(Z) * 25.0,  # Default Z scale (will try to get from metadata)
        "Y": np.arange(Y) * 1.3,  # Default Y scale
        "X": np.arange(X) * 1.3,  # Default X scale
    }

    data_xr = xr.DataArray(
        data_array,
        dims=["T", "C", "Z", "Y", "X"],
        coords=coords,
    )

    return data_xr


# === Data Loading ===
def load_ome_zarr_fov(
    zarr_path: Path | str, fov_path: Path | str, resolution: int = 0
) -> xr.DataArray:
    """
    Load a field of view from an OME-Zarr store as an xarray DataArray.

    Parameters
    ----------
    zarr_path : Path | str
        Path to the root OME-Zarr store
    fov_path : Path | str
        Relative path to the FOV (e.g., "A/1/001007")
    resolution : int, optional
        Resolution level to load (0 is full resolution), by default 0

    Returns
    -------
    xr.DataArray
        Image data with labeled dimensions (T, C, Z, Y, X)
    """
    zarr_path = Path(zarr_path)
    fov_path = Path(fov_path)

    print(f"Loading zarr store from: {zarr_path}")
    print(f"Accessing FOV: {fov_path}")

    # Load as xarray Dataset
    fov_dataset: xr.Dataset = open_ome_dataset(
        zarr_path / fov_path, resolution=resolution, validate=False
    )

    # Extract the image DataArray
    data_xr = fov_dataset["image"]

    print(f"Loaded data shape: {dict(data_xr.sizes)}")
    print(f"Dimensions: {list(data_xr.dims)}")
    print(f"Data type: {data_xr.dtype}")

    return data_xr


# === Image Processing ===
def normalize_for_display(
    img_2d: xr.DataArray,
    percentiles: tuple[float, float] = (1, 99),
    clip_to_uint8: bool = True,
) -> np.ndarray:
    """
    Normalize a 2D microscopy image using percentile clipping.

    Uses robust percentile-based normalization to handle outliers
    common in microscopy data. Works with xarray DataArrays to maintain
    labeled dimensions through the processing pipeline.

    Parameters
    ----------
    img_2d : xr.DataArray
        2D image DataArray to normalize
    percentiles : tuple[float, float], optional
        Lower and upper percentiles for clipping, by default (1, 99)
    clip_to_uint8 : bool, optional
        If True, convert to uint8 (0-255), otherwise keep as float (0-1),
        by default True

    Returns
    -------
    np.ndarray
        Normalized numpy array (uint8 if clip_to_uint8=True, else float32)

    Notes
    -----
    Expects xarray.DataArray input. For raw numpy arrays,
    wrap in xarray first: xr.DataArray(array, dims=["Y", "X"])
    """
    # Calculate percentiles using xarray
    p_low = float(img_2d.quantile(percentiles[0] / 100.0).values)
    p_high = float(img_2d.quantile(percentiles[1] / 100.0).values)

    # Handle edge case: no intensity variation
    if p_high - p_low < 1e-10:
        return np.zeros(img_2d.shape, dtype=np.uint8 if clip_to_uint8 else np.float32)

    # Clip and normalize using xarray operations
    img_clipped = img_2d.clip(min=p_low, max=p_high)
    img_normalized = (img_clipped - p_low) / (p_high - p_low)

    # Convert to numpy array
    result = img_normalized.values

    # Convert to requested output format
    if clip_to_uint8:
        result = (result * 255).astype(np.uint8)
    else:
        result = result.astype(np.float32)

    return result


# === Slice Extraction ===
def extract_2d_slice(
    data_xr: xr.DataArray,
    t: int | None = None,
    c: int | None = None,
    z: int | None = None,
    normalize: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract and optionally normalize a 2D slice from xarray data.

    Flexibly handles different dimension specifications. If a dimension
    index is None, it will be squeezed out if size=1 or raise an error
    if size>1.

    Parameters
    ----------
    data_xr : xr.DataArray
        Image data with dimensions (T, C, Z, Y, X)
    t : int | None, optional
        Timepoint index, by default None
    c : int | None, optional
        Channel index, by default None
    z : int | None, optional
        Z-slice index, by default None
    normalize : bool, optional
        Whether to normalize for display, by default True
    verbose : bool, optional
        Whether to print slice information, by default True

    Returns
    -------
    np.ndarray
        2D numpy array (normalized uint8 if normalize=True, else raw values)

    Raises
    ------
    ValueError
        If result is empty or not 2D after slicing and squeezing
    """
    # Build selection dictionary for indexed dimensions
    sel_dict = {}
    if t is not None:
        sel_dict["T"] = int(t)
    if c is not None:
        sel_dict["C"] = int(c)
    if z is not None:
        sel_dict["Z"] = int(z)

    # Extract slice using xarray's labeled indexing
    slice_xr = data_xr.isel(**sel_dict) if sel_dict else data_xr

    # Compute if Dask-backed (load from disk)
    if hasattr(slice_xr.data, "compute"):
        slice_xr = slice_xr.compute()

    # Squeeze singleton dimensions (e.g., single channel, single Z)
    slice_xr = slice_xr.squeeze()

    # Validation: ensure non-empty result
    if slice_xr.size == 0:
        raise ValueError(
            f"Empty array after slicing. Selection: {sel_dict}, "
            f"Original shape: {data_xr.shape}"
        )

    # Validation: ensure 2D result
    if slice_xr.ndim != 2:
        raise ValueError(
            f"Expected 2D array after slicing, got shape {slice_xr.shape}. "
            f"Selection: {sel_dict}"
        )

    # Verbose output: print slice information
    if verbose:
        sel_str = (
            ", ".join(f"{k}={v}" for k, v in sel_dict.items())
            if sel_dict
            else "full array"
        )
        print(
            f"Extracted slice: {sel_str}, Shape={slice_xr.shape}, "
            f"Range=[{float(slice_xr.min()):.1f}, {float(slice_xr.max()):.1f}]"
        )

    # Normalize or convert to numpy
    if normalize:
        slice_2d = normalize_for_display(slice_xr)
    else:
        slice_2d = slice_xr.values

    return slice_2d


# === Slice Extraction Factory ===
def create_slice_extractor(
    data_xr: xr.DataArray,
    normalize: bool = True,
    channel: int = 0,
):
    """
    Create a closure function for extracting slices from a specific dataset.

    This factory function is useful for Gradio callbacks where the data
    is loaded once and the same extraction function is called multiple times.

    Parameters
    ----------
    data_xr : xr.DataArray
        Image data to extract slices from
    normalize : bool, optional
        Whether to normalize for display, by default True
    channel : int, optional
        Default channel to use, by default 0

    Returns
    -------
    callable
        Function with signature (t: int, z: int) -> np.ndarray that extracts
        and normalizes 2D slices
    """

    def get_slice(t: int, z: int) -> np.ndarray:
        """Extract and normalize a 2D slice at timepoint t and z-slice z."""
        return extract_2d_slice(
            data_xr,
            t=int(t),
            c=channel,
            z=int(z),
            normalize=normalize,
            verbose=True,
        )

    return get_slice


# === Metadata Helpers ===
def get_dimension_info(data_xr: xr.DataArray) -> dict:
    """
    Extract dimension information from xarray DataArray.

    Parameters
    ----------
    data_xr : xr.DataArray
        Image data with dimensions

    Returns
    -------
    dict
        Dictionary with keys: 'sizes', 'dims', 'coords', 'dtype'
    """
    return {
        "sizes": dict(data_xr.sizes),
        "dims": list(data_xr.dims),
        "coords": {dim: data_xr.coords[dim].values.tolist() for dim in data_xr.dims},
        "dtype": str(data_xr.dtype),
    }


def print_data_summary(data_xr: xr.DataArray) -> None:
    """
    Print a formatted summary of xarray DataArray.

    Parameters
    ----------
    data_xr : xr.DataArray
        Image data to summarize
    """
    info = get_dimension_info(data_xr)

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Shape: {info['sizes']}")
    print(f"Dimensions: {info['dims']}")
    print(f"Data type: {info['dtype']}")

    # Print coordinate ranges
    print("\nCoordinate Ranges:")
    for dim in info["dims"]:
        coords = info["coords"][dim]
        if len(coords) > 0:
            print(f"  {dim}: [{coords[0]:.2f} ... {coords[-1]:.2f}] (n={len(coords)})")

    # Print memory size estimate
    total_elements = np.prod(list(info["sizes"].values()))
    dtype_size = np.dtype(data_xr.dtype).itemsize
    size_mb = (total_elements * dtype_size) / (1024**2)
    print(f"\nEstimated size: {size_mb:.1f} MB")
    print("=" * 60 + "\n")


# === Phase Reconstruction Functions ===
def run_reconstruction(zyx_tile: torch.Tensor, recon_args: dict) -> torch.Tensor:
    """
    Run phase reconstruction on a Z-stack.

    Takes a 3D stack (Z, Y, X) and produces a 2D phase reconstruction (Y, X).
    Device is inferred from the input tensor.

    Parameters
    ----------
    zyx_tile : torch.Tensor
        Input Z-stack data with shape (Z, Y, X). Can be on CPU or GPU.
    recon_args : dict
        Reconstruction arguments including wavelength, NA, pixel sizes, etc.
        All tensor values should be on the same device as zyx_tile.

    Returns
    -------
    torch.Tensor
        Reconstructed 2D phase image with shape (Y, X), on same device as input.

    Notes
    -----
    All intermediate tensors are created on the same device as the input
    to ensure efficient computation without device transfers.
    """
    # Infer device from input tensor
    device = zyx_tile.device

    # Prepare transfer function arguments
    tf_args = recon_args.copy()
    Z, _, _ = zyx_tile.shape
    tf_args["z_position_list"] = (
        torch.arange(Z, device=device) - (Z // 2) + recon_args["z_offset"]
    ) * recon_args["z_scale"]
    tf_args.pop("z_offset")
    tf_args.pop("z_scale")

    # Core reconstruction calls (all on same device)
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
    """
    Compute midband power metric for optimization loss.

    Parameters
    ----------
    yx_array : torch.Tensor
        2D reconstructed image (on CPU or GPU)
    NA_det : float
        Numerical aperture of detection
    lambda_ill : float
        Illumination wavelength
    pixel_size : float
        Pixel size in same units as wavelength
    band : tuple[float, float], optional
        Frequency band as fraction of cutoff, by default (0.125, 0.25)

    Returns
    -------
    torch.Tensor
        Scalar power value in the specified frequency band, on same device as input.

    Notes
    -----
    All operations are performed on the same device as the input tensor
    for efficient GPU computation.
    """
    device = yx_array.device

    # Generate frequency coordinates (returns numpy arrays)
    _, _, fxx, fyy = util.gen_coordinate(yx_array.shape, pixel_size)

    # Convert to torch tensor on same device
    frr = torch.tensor(np.sqrt(fxx**2 + fyy**2), dtype=torch.float32, device=device)

    # FFT and frequency masking (all on device)
    xy_abs_fft = torch.abs(torch.fft.fftn(yx_array))
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(frr > cutoff * band[0], frr < cutoff * band[1])

    return torch.sum(xy_abs_fft[mask])


def prepare_optimizer(
    optimizable_params: dict[str, tuple[bool, float, float]],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Parameter], torch.optim.Optimizer]:
    """
    Prepare optimization parameters and Adam optimizer.

    Parameters
    ----------
    optimizable_params : dict
        Dict mapping param names to (enabled, initial_value, learning_rate)
    device : torch.device
        Device to create parameters on (CPU or GPU)

    Returns
    -------
    tuple[dict, Optimizer]
        optimization_params dict and configured optimizer

    Notes
    -----
    All parameters are created on the specified device for efficient
    GPU-accelerated optimization if available.
    """
    optimization_params: dict[str, torch.nn.Parameter] = {}
    optimizer_config = []
    for name, (enabled, initial, lr) in optimizable_params.items():
        if enabled:
            param = torch.nn.Parameter(
                torch.tensor([initial], dtype=torch.float32, device=device),
                requires_grad=True,
            )
            optimization_params[name] = param
            optimizer_config.append({"params": [param], "lr": lr})

    optimizer = torch.optim.Adam(optimizer_config)
    return optimization_params, optimizer


def run_reconstruction_single(
    zyx_stack: np.ndarray,
    pixel_scales: tuple[float, float, float],
    fixed_params: dict,
    param_values: dict,
    device: Device = None,
) -> np.ndarray:
    """
    Run a single phase reconstruction with specified parameters (no optimization).

    Parameters
    ----------
    zyx_stack : np.ndarray
        Input Z-stack with shape (Z, Y, X)
    pixel_scales : tuple[float, float, float]
        (z_scale, y_scale, x_scale) in micrometers
    fixed_params : dict
        Fixed reconstruction parameters (wavelength, index, etc.)
    param_values : dict
        Parameter values to use (z_offset, numerical_aperture_detection, etc.)
    device : torch.device | str | None, optional
        Computing device. If None, auto-selects GPU if available, else CPU.

    Returns
    -------
    np.ndarray
        Normalized uint8 array of reconstructed phase image (for display)
    """
    # Resolve device (will print GPU info if available)
    device = get_device(device)

    # Convert to torch tensor on target device
    zyx_tile = torch.tensor(zyx_stack, dtype=torch.float32, device=device)

    # Prepare reconstruction arguments
    z_scale, y_scale, x_scale = pixel_scales
    recon_args = fixed_params.copy()

    # Remove non-reconstruction parameters from fixed_params
    recon_args.pop("num_iterations", None)
    recon_args.pop("use_tiling", None)
    recon_args.pop("device", None)

    recon_args["yx_shape"] = zyx_tile.shape[1:]
    recon_args["yx_pixel_size"] = y_scale
    recon_args["z_scale"] = z_scale

    # Set parameter values (convert to tensors on device)
    for name, value in param_values.items():
        recon_args[name] = torch.tensor([value], dtype=torch.float32, device=device)

    # Run reconstruction
    yx_recon = run_reconstruction(zyx_tile, recon_args)

    # Transfer to CPU and normalize for display
    recon_numpy = yx_recon.detach().cpu().numpy()
    # Wrap in xarray for normalize_for_display (expects xr.DataArray)
    recon_normalized = normalize_for_display(xr.DataArray(recon_numpy))

    return recon_normalized


def run_optimization_streaming(
    zyx_stack: np.ndarray,
    pixel_scales: tuple[float, float, float],
    fixed_params: dict,
    optimizable_params: dict,
    num_iterations: int = 10,
    device: Device = None,
) -> Generator[dict, None, None]:
    """
    Run phase reconstruction optimization with streaming updates.

    Generator that yields reconstruction results and loss after each iteration.
    Supports GPU acceleration for significant speedup (15-25x on typical hardware).

    Parameters
    ----------
    zyx_stack : np.ndarray
        Input Z-stack with shape (Z, Y, X)
    pixel_scales : tuple[float, float, float]
        (z_scale, y_scale, x_scale) in micrometers
    fixed_params : dict
        Fixed reconstruction parameters (wavelength, index, etc.)
    optimizable_params : dict
        Parameters to optimize with (enabled, initial, lr) tuples
    num_iterations : int, optional
        Number of optimization iterations, by default 10
    device : torch.device | str | None, optional
        Computing device. If None, auto-selects GPU if available, else CPU.
        Examples: "cuda", "cpu", "cuda:0", torch.device("cuda")
        By default None

    Yields
    ------
    dict
        Dictionary with keys:
        - 'reconstructed_image': normalized uint8 array (on CPU for display)
        - 'loss': float loss value
        - 'iteration': int iteration number (1-indexed)
        - 'params': dict of current parameter values

    Notes
    -----
    All computation is performed on the specified device (GPU if available).
    Only final results are transferred to CPU for display, minimizing
    transfer overhead.
    """
    # Resolve device (will print GPU info if available)
    device = get_device(device)

    # Convert to torch tensor on target device (single transfer)
    zyx_tile = torch.tensor(zyx_stack, dtype=torch.float32, device=device)

    # Prepare reconstruction arguments
    z_scale, y_scale, x_scale = pixel_scales
    recon_args = fixed_params.copy()

    # Remove non-reconstruction parameters from fixed_params
    recon_args.pop("num_iterations", None)
    recon_args.pop("use_tiling", None)
    recon_args.pop("device", None)  # Remove device if present

    recon_args["yx_shape"] = zyx_tile.shape[1:]
    recon_args["yx_pixel_size"] = y_scale
    recon_args["z_scale"] = z_scale

    # Initialize optimizable parameters on device
    for name, (enabled, initial, lr) in optimizable_params.items():
        recon_args[name] = torch.tensor([initial], dtype=torch.float32, device=device)

    # Prepare optimizer with parameters on device
    optimization_params, optimizer = prepare_optimizer(optimizable_params, device)

    # Optimization loop (all on device)
    for step in range(num_iterations):
        # Update parameters
        for name, param in optimization_params.items():
            recon_args[name] = param

        # Run reconstruction (all on device)
        yx_recon = run_reconstruction(zyx_tile, recon_args)

        # Compute loss (all on device, negative midband power - we want to maximize)
        loss = -compute_midband_power(
            yx_recon,
            NA_det=0.15,
            lambda_ill=recon_args["wavelength_illumination"],
            pixel_size=recon_args["yx_pixel_size"],
            band=(0.1, 0.2),
        )

        # Backward pass and optimizer step (on device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Transfer to CPU ONLY for display (single transfer per iteration)
        recon_numpy = yx_recon.detach().cpu().numpy()
        # Wrap in xarray for normalize_for_display (expects xr.DataArray)
        recon_normalized = normalize_for_display(xr.DataArray(recon_numpy))

        # Extract current parameter values (scalars, already on CPU)
        param_values = {
            name: param.item() for name, param in optimization_params.items()
        }

        # Yield results
        yield {
            "reconstructed_image": recon_normalized,
            "loss": loss.item(),
            "iteration": step + 1,
            "params": param_values,
        }


def extract_tiles(
    zyx_data: np.ndarray, num_tiles: tuple[int, int], overlap_pct: float
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int, int]]]:
    """
    Extract overlapping tiles from a Z-stack for processing.

    Parameters
    ----------
    zyx_data : np.ndarray
        Input data with shape (Z, Y, X)
    num_tiles : tuple[int, int]
        Number of tiles in (Y, X) dimensions
    overlap_pct : float
        Overlap percentage between tiles (0.0 to 1.0)

    Returns
    -------
    tuple[dict, dict]
        tiles: dict mapping tile names to arrays
        translations: dict mapping tile names to (z, y, x) positions
    """
    Z, Y, X = zyx_data.shape
    tile_height = int(np.ceil(Y / (num_tiles[0] - (num_tiles[0] - 1) * overlap_pct)))
    tile_width = int(np.ceil(X / (num_tiles[1] - (num_tiles[1] - 1) * overlap_pct)))
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
