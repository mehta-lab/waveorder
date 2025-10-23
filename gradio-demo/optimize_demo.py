"""
Gradio Phase Reconstruction Viewer

Interactive web interface for viewing zarr microscopy data with T/Z navigation.
Based on: docs/examples/visuals/optimize_phase_recon.py
"""

import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path

from demo_utils import (
    print_data_summary,
    run_optimization_streaming,
    get_plate_metadata,
    load_fov_from_plate,
)

# === Configuration ===
INPUT_PATH = Path(
    "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/0-convert/live_imaging/tracking_symlink.zarr"
)

# Default FOV (can be changed via UI)
DEFAULT_ROW = "A"
DEFAULT_COLUMN = "1"
DEFAULT_FIELD = "001007"

# Reconstruction configuration
RECON_CONFIG = {
    "wavelength_illumination": 0.450,
    "index_of_refraction_media": 1.0,
    "invert_phase_contrast": True,
    "num_iterations": 10,
    # GPU Configuration (auto-detects GPU for 15-25x speedup)
    # - None: Auto-detect (uses CUDA if available, else CPU)
    # - "cuda": Force GPU usage (requires CUDA-capable device)
    # - "cpu": Force CPU usage (for testing/debugging)
    "device": None,
    # Tiling (not implemented - using full image)
    "use_tiling": False,
}

OPTIMIZABLE_PARAMS = {
    "z_offset": (True, 0.0, 0.01),
    "numerical_aperture_detection": (True, 0.15, 0.001),
    "numerical_aperture_illumination": (True, 0.1, 0.001),
    "tilt_angle_zenith": (True, 0.1, 0.005),
    "tilt_angle_azimuth": (True, 260 * np.pi / 180, 0.001),
}

# === Load Plate Metadata (Fast) ===
print("\n" + "=" * 60)
print("Loading HCS Plate Metadata...")
print("=" * 60)
plate_metadata = get_plate_metadata(INPUT_PATH)

print(f"Available rows: {plate_metadata['rows']}")
print(f"Available columns: {plate_metadata['columns']}")
print(f"Total wells: {len(plate_metadata['wells'])}")

# Get default well fields
default_well_key = (DEFAULT_ROW, DEFAULT_COLUMN)
default_fields = plate_metadata["wells"].get(default_well_key, [])
print(f"Fields in {DEFAULT_ROW}/{DEFAULT_COLUMN}: {len(default_fields)}")
print("=" * 60 + "\n")

# Load default FOV
print(f"Loading default FOV: {DEFAULT_ROW}/{DEFAULT_COLUMN}/{DEFAULT_FIELD}")
data_xr = load_fov_from_plate(
    plate_metadata["plate"], DEFAULT_ROW, DEFAULT_COLUMN, DEFAULT_FIELD, resolution=0
)
print_data_summary(data_xr)

# Extract pixel scales for reconstruction
pixel_scales = (
    float(data_xr.coords["Z"][1] - data_xr.coords["Z"][0]),  # z_scale
    float(data_xr.coords["Y"][1] - data_xr.coords["Y"][0]),  # y_scale
    float(data_xr.coords["X"][1] - data_xr.coords["X"][0]),  # x_scale
)
print(f"Pixel scales (Z, Y, X): {pixel_scales} micrometers")


# === Gradio Interface ===
with gr.Blocks(
    title="Phase Reconstruction Viewer",
    theme=gr.themes.Default(),
    css="""
        /* Make ImageSlider fill container and maintain aspect ratio */
        .image-container img {
            object-fit: contain !important;
            max-height: 800px !important;
        }
        /* Ensure image slider takes full width */
        .image-frame {
            width: 100% !important;
        }
    """,
) as demo:
    gr.Markdown("# Phase Reconstruction Viewer")
    gr.Markdown(f"**Data Path:** `{INPUT_PATH}`")
    gr.Markdown("---")

    # Two-column layout: Image viewer (left) | Controls (right)
    with gr.Row():
        # LEFT COLUMN: Large ImageSlider (60% width)
        with gr.Column(scale=4):
            # Section 1: FOV Selection (above image)
            gr.Markdown("### 📂 Select Field of View")
            with gr.Row():
                row_dropdown = gr.Dropdown(
                    choices=plate_metadata["rows"],
                    value=DEFAULT_ROW,
                    label="Row",
                    interactive=True,
                    scale=1,
                )
                column_dropdown = gr.Dropdown(
                    choices=plate_metadata["columns"],
                    value=DEFAULT_COLUMN,
                    label="Column",
                    interactive=True,
                    scale=1,
                )
                field_dropdown = gr.Dropdown(
                    choices=default_fields,
                    value=DEFAULT_FIELD,
                    label="Field",
                    interactive=True,
                    scale=2,
                )
                load_fov_btn = gr.Button(
                    "🔄 Load", variant="secondary", size="sm", scale=1
                )

            gr.Markdown("---")

            # Image viewer
            from demo_utils import extract_2d_slice

            initial_preview = extract_2d_slice(
                data_xr,
                t=0,
                c=0,
                z=data_xr.sizes["Z"] // 2,
                normalize=True,
                verbose=False,
            )

            image_viewer = gr.ImageSlider(
                label="Raw (left) vs Reconstructed (right) - Drag slider to compare",
                type="numpy",
                value=(initial_preview, initial_preview),
                height=800,  # Large display
            )

            gr.Markdown("---")

            # Section 2: Navigation (below image)
            gr.Markdown("### 🎛️ Navigation")
            t_slider = gr.Slider(
                minimum=0,
                maximum=data_xr.sizes["T"] - 1,
                value=0,
                step=1,
                label="Timepoint (T)",
                scale=1,
            )
            z_slider = gr.Slider(
                minimum=0,
                maximum=data_xr.sizes["Z"] - 1,
                value=data_xr.sizes["Z"] // 2,
                step=1,
                label="Z-slice",
                scale=1,
            )

        # RIGHT COLUMN: All controls (40% width)
        with gr.Column(scale=2):
            # Section 3: Reconstruction Parameters
            gr.Markdown("### ⚙️ Reconstruction Parameters")

            # Sliders for optimizable parameters
            z_offset_slider = gr.Slider(
                minimum=-0.5,
                maximum=0.5,
                value=OPTIMIZABLE_PARAMS["z_offset"][1],
                step=0.01,
                label="Z Offset (μm)",
                info="Axial focus offset",
            )

            na_det_slider = gr.Slider(
                minimum=0.05,
                maximum=0.4,
                value=OPTIMIZABLE_PARAMS["numerical_aperture_detection"][1],
                step=0.001,
                label="NA Detection",
                info="Numerical aperture of detection objective",
            )

            na_ill_slider = gr.Slider(
                minimum=0.05,
                maximum=0.3,
                value=OPTIMIZABLE_PARAMS["numerical_aperture_illumination"][1],
                step=0.001,
                label="NA Illumination",
                info="Numerical aperture of illumination",
            )

            tilt_zenith_slider = gr.Slider(
                minimum=0.0,
                maximum=np.pi / 2,
                value=OPTIMIZABLE_PARAMS["tilt_angle_zenith"][1],
                step=0.005,
                label="Tilt Zenith (rad)",
                info="Zenith angle of illumination tilt",
            )

            tilt_azimuth_slider = gr.Slider(
                minimum=0.0,
                maximum=2 * np.pi,
                value=OPTIMIZABLE_PARAMS["tilt_angle_azimuth"][1],
                step=0.001,
                label="Tilt Azimuth (rad)",
                info="Azimuthal angle of illumination tilt",
            )

            gr.Markdown("---")

            # Section 4: Reconstruction Actions
            gr.Markdown("### 🔬 Phase Reconstruction")

            with gr.Row():
                optimize_btn = gr.Button(
                    "⚡ Optimize Parameters", variant="secondary", size="lg"
                )
                reconstruct_btn = gr.Button(
                    "🔬 Run Reconstruction", variant="primary", size="lg"
                )

            gr.Markdown("---")

            # Section 5: Optimization Results
            gr.Markdown("### 📊 Optimization Results")

            loss_plot = gr.LinePlot(
                x="iteration",
                y="loss",
                title="Optimization - Midband Spatial Frequency Loss",
                height=200,
                scale=2,
            )

            # Iteration scrubbing controls
            iteration_slider = gr.Slider(
                minimum=1,
                maximum=1,
                value=1,
                step=1,
                label="View Iteration",
                info="Scrub through optimization history",
                interactive=False,
                visible=False,
            )
            iteration_info = gr.Markdown(
                value="",
                visible=False,
            )

    # State storage
    iteration_history = gr.State(value=[])
    current_data_xr = gr.State(value=data_xr)
    current_pixel_scales = gr.State(value=pixel_scales)

    gr.Markdown("---")
    gr.Markdown(
        "💡 **Workflow:** Select FOV → Navigate with sliders → Click Reconstruct → View optimized results"
    )

    # === FOV Selection Callbacks ===
    def update_field_dropdown(row: str, column: str):
        """Update available fields when row/column changes."""
        well_key = (row, column)
        fields = plate_metadata["wells"].get(well_key, [])
        return gr.Dropdown(choices=fields, value=fields[0] if fields else None)

    def load_selected_fov(
        row: str, column: str, field: str, current_t: int, current_z: int
    ):
        """Load selected FOV and update UI components."""
        try:
            print(f"\nLoading FOV: {row}/{column}/{field}")

            # Load new data
            new_data_xr = load_fov_from_plate(
                plate_metadata["plate"], row, column, field, resolution=0
            )

            # Calculate pixel scales
            new_pixel_scales = (
                float(new_data_xr.coords["Z"][1] - new_data_xr.coords["Z"][0]),
                float(new_data_xr.coords["Y"][1] - new_data_xr.coords["Y"][0]),
                float(new_data_xr.coords["X"][1] - new_data_xr.coords["X"][0]),
            )

            # Update sliders
            t_max = new_data_xr.sizes["T"] - 1
            z_max = new_data_xr.sizes["Z"] - 1

            # Clamp current values to new ranges
            new_t = min(current_t, t_max)
            new_z = min(current_z, z_max)

            print(
                f"✅ Loaded: T={new_data_xr.sizes['T']}, Z={new_data_xr.sizes['Z']}, Y={new_data_xr.sizes['Y']}, X={new_data_xr.sizes['X']}"
            )

            # Get preview image (show same image twice for preview mode)
            from demo_utils import extract_2d_slice

            preview_image = extract_2d_slice(
                new_data_xr, t=new_t, c=0, z=new_z, normalize=True, verbose=False
            )

            return (
                gr.Slider(maximum=t_max, value=new_t),  # Updated T slider
                gr.Slider(maximum=z_max, value=new_z),  # Updated Z slider
                (preview_image, preview_image),  # ImageSlider in preview mode
                new_data_xr,  # Update state
                new_pixel_scales,  # Update state
            )

        except Exception as e:
            error_msg = f"❌ Error loading FOV: {str(e)}"
            print(error_msg)
            return (
                gr.skip(),  # Keep T slider
                gr.skip(),  # Keep Z slider
                gr.skip(),  # Keep image viewer
                gr.skip(),  # Keep data state
                gr.skip(),  # Keep pixel_scales state
            )

    # === Image Display Callbacks ===
    def get_slice_for_preview(t: int, z: int, data_xr_state):
        """Extract slice and show in preview mode (same image twice)."""
        from demo_utils import extract_2d_slice

        slice_img = extract_2d_slice(
            data_xr_state, t=int(t), c=0, z=int(z), normalize=True, verbose=False
        )
        return (slice_img, slice_img)  # Preview mode: both sides show same image

    # Wire FOV selection
    row_dropdown.change(
        fn=update_field_dropdown,
        inputs=[row_dropdown, column_dropdown],
        outputs=[field_dropdown],
    )
    column_dropdown.change(
        fn=update_field_dropdown,
        inputs=[row_dropdown, column_dropdown],
        outputs=[field_dropdown],
    )

    load_fov_btn.click(
        fn=load_selected_fov,
        inputs=[row_dropdown, column_dropdown, field_dropdown, t_slider, z_slider],
        outputs=[
            t_slider,
            z_slider,
            image_viewer,
            current_data_xr,
            current_pixel_scales,
        ],
    )

    # Wire image viewer for T/Z navigation (preview mode: same image twice)
    demo.load(
        fn=get_slice_for_preview,
        inputs=[t_slider, z_slider, current_data_xr],
        outputs=image_viewer,
    )
    t_slider.change(
        fn=get_slice_for_preview,
        inputs=[t_slider, z_slider, current_data_xr],
        outputs=image_viewer,
    )
    z_slider.change(
        fn=get_slice_for_preview,
        inputs=[t_slider, z_slider, current_data_xr],
        outputs=image_viewer,
    )

    # Wire reconstruction button
    def run_reconstruction_ui(
        t: int,
        z: int,
        z_offset: float,
        na_det: float,
        na_ill: float,
        tilt_zenith: float,
        tilt_azimuth: float,
        data_xr_state,
        pixel_scales_state,
    ):
        """
        Run reconstruction with CURRENT slider values (no optimization).

        Uses slider parameters directly for a single fast reconstruction.
        """
        from demo_utils import extract_2d_slice, run_reconstruction_single

        # Extract full Z-stack for timepoint t (for reconstruction)
        zyx_stack = data_xr_state.isel(T=int(t), C=0).values

        # Get current Z-slice for comparison (left side of ImageSlider)
        original_normalized = extract_2d_slice(
            data_xr_state, t=int(t), c=0, z=int(z), normalize=True, verbose=False
        )

        # Build parameter dict from slider values
        param_values = {
            "z_offset": z_offset,
            "numerical_aperture_detection": na_det,
            "numerical_aperture_illumination": na_ill,
            "tilt_angle_zenith": tilt_zenith,
            "tilt_angle_azimuth": tilt_azimuth,
        }

        # Run single reconstruction with these parameters
        reconstructed_image = run_reconstruction_single(
            zyx_stack, pixel_scales_state, RECON_CONFIG, param_values
        )

        # Return updated image slider (no optimization results)
        return (original_normalized, reconstructed_image)

    def run_optimization_ui(t: int, z: int, data_xr_state, pixel_scales_state):
        """
        Run OPTIMIZATION and stream updates to UI with iteration caching.

        Uses OPTIMIZABLE_PARAMS as initial guesses, runs full optimization loop.
        Yields progressive updates for ImageSlider, loss plot, status,
        iteration history, iteration slider, and SLIDER UPDATES.
        """
        # Extract full Z-stack for timepoint t (for reconstruction)
        zyx_stack = data_xr_state.isel(T=int(t), C=0).values

        # Get current Z-slice for comparison (left side of ImageSlider)
        from demo_utils import extract_2d_slice

        original_normalized = extract_2d_slice(
            data_xr_state, t=int(t), c=0, z=int(z), normalize=True, verbose=False
        )

        # Initialize tracking
        loss_history = []
        iteration_cache = []

        # Set raw image once at the start (pin it)
        yield (
            (
                original_normalized,
                original_normalized,
            ),  # Show raw image on both sides initially
            gr.skip(),  # Don't update loss plot yet
            [],  # Clear iteration history
            gr.Slider(visible=False, interactive=False),  # Hide iteration slider
            gr.Markdown(value="Starting optimization...", visible=True),
            # Slider updates (11 outputs total):
            gr.skip(),  # z_offset
            gr.skip(),  # na_det
            gr.skip(),  # na_ill
            gr.skip(),  # tilt_zenith
            gr.skip(),  # tilt_azimuth
        )

        # Run optimization with streaming
        for result in run_optimization_streaming(
            zyx_stack,
            pixel_scales_state,
            RECON_CONFIG,
            OPTIMIZABLE_PARAMS,
            num_iterations=RECON_CONFIG["num_iterations"],
        ):
            # Current iteration number
            n = result["iteration"]

            # Cache iteration result
            iteration_cache.append(
                {
                    "iteration": n,
                    "reconstructed_image": result["reconstructed_image"],
                    "loss": result["loss"],
                    "params": result["params"],
                    "raw_image": original_normalized,
                }
            )

            # Accumulate loss history (ensure iteration is int for proper x-axis)
            loss_history.append(
                {
                    "iteration": int(n),
                    "loss": result["loss"],
                }
            )

            # Format iteration info
            info_md = f"**Iteration {n}/{RECON_CONFIG['num_iterations']}** | Loss: `{result['loss']:.2e}`"

            # Yield updates - update ImageSlider AND sliders with latest params
            yield (
                (
                    original_normalized,
                    result["reconstructed_image"],
                ),  # Update ImageSlider
                pd.DataFrame(loss_history),  # Loss plot
                iteration_cache,  # Update iteration history state
                gr.Slider(  # Update iteration slider (grows from 1-1 to 1-10)
                    minimum=1,
                    maximum=n,
                    value=n,
                    step=1,
                    visible=True,
                    interactive=True,
                ),
                gr.Markdown(value=info_md, visible=True),  # Show iteration info
                # Update parameter sliders with optimized values:
                result["params"].get("z_offset", gr.skip()),
                result["params"].get("numerical_aperture_detection", gr.skip()),
                result["params"].get("numerical_aperture_illumination", gr.skip()),
                result["params"].get("tilt_angle_zenith", gr.skip()),
                result["params"].get("tilt_angle_azimuth", gr.skip()),
            )

        # Final yield (keep last state)
        yield (
            gr.skip(),  # Keep last ImageSlider state
            gr.skip(),  # Keep last loss plot
            gr.skip(),  # Keep iteration history
            gr.skip(),  # Keep iteration slider
            gr.Markdown(
                value=f"**Optimization Complete!** Final Loss: `{result['loss']:.2e}`",
                visible=True,
            ),
            gr.skip(),  # Keep z_offset
            gr.skip(),  # Keep na_det
            gr.skip(),  # Keep na_ill
            gr.skip(),  # Keep tilt_zenith
            gr.skip(),  # Keep tilt_azimuth
        )

    # Iteration scrubbing callback
    def scrub_iterations(iteration_idx: int, history: list):
        """Update display when user scrubs to different iteration."""

        if not history or iteration_idx < 1 or iteration_idx > len(history):
            return gr.skip(), gr.skip()

        # Get selected iteration (convert to 0-indexed)
        selected = history[iteration_idx - 1]

        # Update ImageSlider overlay
        comparison = (selected["raw_image"], selected["reconstructed_image"])

        # Update info display
        info_md = f"**Iteration {selected['iteration']}/{len(history)}** | Loss: `{selected['loss']:.2e}`"

        return comparison, info_md

    # Context change handler - clear iteration state when T/Z changes
    def clear_iteration_state():
        """Reset iteration state when coordinates change."""
        return (
            [],  # Clear iteration_history
            gr.Slider(
                visible=False, value=1, maximum=1, interactive=False
            ),  # Hide slider
            gr.Markdown(value="", visible=False),  # Hide info
        )

    # Wire optimize button (runs full optimization loop, updates sliders)
    optimize_btn.click(
        fn=run_optimization_ui,
        inputs=[t_slider, z_slider, current_data_xr, current_pixel_scales],
        outputs=[
            image_viewer,
            loss_plot,
            iteration_history,
            iteration_slider,
            iteration_info,
            z_offset_slider,
            na_det_slider,
            na_ill_slider,
            tilt_zenith_slider,
            tilt_azimuth_slider,
        ],
    )

    # Wire reconstruction button (uses current slider values, fast single reconstruction)
    reconstruct_btn.click(
        fn=run_reconstruction_ui,
        inputs=[
            t_slider,
            z_slider,
            z_offset_slider,
            na_det_slider,
            na_ill_slider,
            tilt_zenith_slider,
            tilt_azimuth_slider,
            current_data_xr,
            current_pixel_scales,
        ],
        outputs=[image_viewer],
    )

    # Wire iteration scrubbing
    iteration_slider.change(
        fn=scrub_iterations,
        inputs=[iteration_slider, iteration_history],
        outputs=[image_viewer, iteration_info],
    )

    # Wire T/Z slider changes to clear iteration state
    t_slider.change(
        fn=clear_iteration_state,
        inputs=[],
        outputs=[iteration_history, iteration_slider, iteration_info],
    )
    z_slider.change(
        fn=clear_iteration_state,
        inputs=[],
        outputs=[iteration_history, iteration_slider, iteration_info],
    )


# === Launch ===
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Gradio Phase Reconstruction Viewer")
    print("=" * 60)
    print("Open your browser to the URL shown below")
    print("=" * 60 + "\n")

    demo.launch(
        share=False,  # Set to True to create public link
        # server_name="0.0.0.0",  # Allow external access
        server_port=7860,
    )
