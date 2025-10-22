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
print("\n" + "="*60)
print("Loading HCS Plate Metadata...")
print("="*60)
plate_metadata = get_plate_metadata(INPUT_PATH)

print(f"Available rows: {plate_metadata['rows']}")
print(f"Available columns: {plate_metadata['columns']}")
print(f"Total wells: {len(plate_metadata['wells'])}")

# Get default well fields
default_well_key = (DEFAULT_ROW, DEFAULT_COLUMN)
default_fields = plate_metadata['wells'].get(default_well_key, [])
print(f"Fields in {DEFAULT_ROW}/{DEFAULT_COLUMN}: {len(default_fields)}")
print("="*60 + "\n")

# Load default FOV
print(f"Loading default FOV: {DEFAULT_ROW}/{DEFAULT_COLUMN}/{DEFAULT_FIELD}")
data_xr = load_fov_from_plate(
    plate_metadata['plate'],
    DEFAULT_ROW,
    DEFAULT_COLUMN,
    DEFAULT_FIELD,
    resolution=0
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
) as demo:
    gr.Markdown("# Phase Reconstruction Viewer")
    gr.Markdown(f"**Data Path:** `{INPUT_PATH}`")

    # FOV Selection Controls
    with gr.Accordion("📂 Select Field of View", open=True):
        with gr.Row():
            row_dropdown = gr.Dropdown(
                choices=plate_metadata['rows'],
                value=DEFAULT_ROW,
                label="Row",
                interactive=True,
            )
            column_dropdown = gr.Dropdown(
                choices=plate_metadata['columns'],
                value=DEFAULT_COLUMN,
                label="Column",
                interactive=True,
            )
            field_dropdown = gr.Dropdown(
                choices=default_fields,
                value=DEFAULT_FIELD,
                label="Field",
                interactive=True,
            )
        load_fov_btn = gr.Button("🔄 Load Selected FOV", variant="primary")
        fov_status = gr.Markdown(
            f"**Current FOV:** {DEFAULT_ROW}/{DEFAULT_COLUMN}/{DEFAULT_FIELD} | "
            f"**Shape:** {dict(data_xr.sizes)} | **Dims:** {list(data_xr.dims)}"
        )

    gr.Markdown("---")

    # Side-by-side layout: Raw viewer (left) | Reconstruction viewer (right)
    with gr.Row():
        # LEFT: Raw data viewer
        with gr.Column(scale=1):
            gr.Markdown("### Raw Data Viewer")
            raw_image = gr.Image(
                label="Raw Microscopy Data",
                type="numpy",
            )
            t_slider = gr.Slider(
                minimum=0,
                maximum=data_xr.sizes["T"] - 1,
                value=0,
                step=1,
                label="Timepoint (T)",
            )
            z_slider = gr.Slider(
                minimum=0,
                maximum=data_xr.sizes["Z"] - 1,
                value=data_xr.sizes["Z"] // 2,
                step=1,
                label="Z-slice",
            )

        # RIGHT: Reconstruction viewer
        with gr.Column(scale=1):
            gr.Markdown("### Phase Reconstruction")
            comparison_slider = gr.ImageSlider(
                label="Raw vs Reconstructed",
                type="numpy",
            )

            # Iteration scrubbing controls
            with gr.Group():
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

            loss_plot = gr.LinePlot(
                x="iteration",
                y="loss",
                title="Optimization Loss",
                width=400,
                height=200,
            )
            reconstruct_btn = gr.Button("🔬 Reconstruct", variant="primary", size="lg")
            status_text = gr.Textbox(
                label="Status",
                value="Ready to reconstruct",
                interactive=False,
            )

    # State storage
    iteration_history = gr.State(value=[])
    current_data_xr = gr.State(value=data_xr)
    current_pixel_scales = gr.State(value=pixel_scales)

    gr.Markdown("---")
    gr.Markdown(
        "💡 **Tip:** Select FOV, navigate with T/Z sliders, then click **Reconstruct** to optimize parameters"
    )

    # === FOV Selection Callbacks ===
    def update_field_dropdown(row: str, column: str):
        """Update available fields when row/column changes."""
        well_key = (row, column)
        fields = plate_metadata['wells'].get(well_key, [])
        return gr.Dropdown(choices=fields, value=fields[0] if fields else None)

    def load_selected_fov(row: str, column: str, field: str, current_t: int, current_z: int):
        """Load selected FOV and update UI components."""
        try:
            print(f"\nLoading FOV: {row}/{column}/{field}")

            # Load new data
            new_data_xr = load_fov_from_plate(
                plate_metadata['plate'],
                row,
                column,
                field,
                resolution=0
            )

            # Calculate pixel scales
            new_pixel_scales = (
                float(new_data_xr.coords["Z"][1] - new_data_xr.coords["Z"][0]),
                float(new_data_xr.coords["Y"][1] - new_data_xr.coords["Y"][0]),
                float(new_data_xr.coords["X"][1] - new_data_xr.coords["X"][0]),
            )

            # Update status
            status = (
                f"**Current FOV:** {row}/{column}/{field} | "
                f"**Shape:** {dict(new_data_xr.sizes)} | **Dims:** {list(new_data_xr.dims)}"
            )

            # Update sliders
            t_max = new_data_xr.sizes["T"] - 1
            z_max = new_data_xr.sizes["Z"] - 1
            z_mid = new_data_xr.sizes["Z"] // 2

            # Clamp current values to new ranges
            new_t = min(current_t, t_max)
            new_z = min(current_z, z_max)

            print(f"✅ Loaded: T={new_data_xr.sizes['T']}, Z={new_data_xr.sizes['Z']}, Y={new_data_xr.sizes['Y']}, X={new_data_xr.sizes['X']}")

            # Get initial preview image
            from demo_utils import extract_2d_slice
            preview_image = extract_2d_slice(new_data_xr, t=new_t, c=0, z=new_z, normalize=True, verbose=False)

            return (
                status,
                gr.Slider(maximum=t_max, value=new_t),  # Updated T slider
                gr.Slider(maximum=z_max, value=new_z),  # Updated Z slider
                new_data_xr,  # Update state
                new_pixel_scales,  # Update state
                preview_image,  # Update raw image display
            )

        except Exception as e:
            error_msg = f"❌ Error loading FOV: {str(e)}"
            print(error_msg)
            return (
                error_msg,
                gr.skip(),  # Keep T slider
                gr.skip(),  # Keep Z slider
                gr.skip(),  # Keep data state
                gr.skip(),  # Keep pixel_scales state
                gr.skip(),  # Keep raw image
            )

    # === Image Display Callbacks ===
    def get_slice_from_state(t: int, z: int, data_xr_state):
        """Extract slice from state data."""
        from demo_utils import extract_2d_slice
        return extract_2d_slice(data_xr_state, t=int(t), c=0, z=int(z), normalize=True, verbose=True)

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
        outputs=[fov_status, t_slider, z_slider, current_data_xr, current_pixel_scales, raw_image],
    )

    # Wire raw image viewer (using state)
    demo.load(fn=get_slice_from_state, inputs=[t_slider, z_slider, current_data_xr], outputs=raw_image)
    t_slider.change(fn=get_slice_from_state, inputs=[t_slider, z_slider, current_data_xr], outputs=raw_image)
    z_slider.change(fn=get_slice_from_state, inputs=[t_slider, z_slider, current_data_xr], outputs=raw_image)

    # Wire reconstruction button
    def run_reconstruction_ui(t: int, z: int, data_xr_state, pixel_scales_state):
        """
        Run optimization and stream updates to UI with iteration caching.

        Yields progressive updates for ImageSlider, loss plot, status,
        iteration history, and iteration slider.
        """
        # Extract full Z-stack for timepoint t (for reconstruction)
        zyx_stack = data_xr_state.isel(T=int(t), C=0).values

        # Get current Z-slice for comparison (left side of ImageSlider)
        from demo_utils import extract_2d_slice
        original_normalized = extract_2d_slice(data_xr_state, t=int(t), c=0, z=int(z), normalize=True, verbose=False)

        # Initialize tracking
        loss_history = []
        iteration_cache = []

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

            # Format parameter values for display
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in result["params"].items())

            # Format iteration info
            info_md = f"**Iteration {n}/{RECON_CONFIG['num_iterations']}** | Loss: `{result['loss']:.2e}`"

            # Yield updates for all output components
            yield (
                (original_normalized, result["reconstructed_image"]),  # ImageSlider
                pd.DataFrame(loss_history),  # Loss plot
                f"Iteration {n}/{RECON_CONFIG['num_iterations']} | "
                f"Loss: {result['loss']:.2e}\n{param_str}",  # Status text
                iteration_cache,  # Update iteration history state
                gr.Slider(  # Update iteration slider (grows from 1-1 to 1-10)
                    minimum=1,
                    maximum=n,  # Grows with each iteration!
                    value=n,  # Tracks latest by default
                    step=1,
                    visible=True,  # Becomes visible on first iteration
                    interactive=True,  # User can scrub while running
                ),
                gr.Markdown(value=info_md, visible=True),  # Show iteration info
            )

        # Final status update
        final_status = (
            f"✅ Optimization complete! Final loss: {loss_history[-1]['loss']:.2e}\n"
            f"🎚️ Use iteration slider to explore optimization history (1-{len(iteration_cache)})"
        )

        yield (
            gr.skip(),  # Keep last ImageSlider state
            gr.skip(),  # Keep last loss plot
            final_status,  # Updated status
            gr.skip(),  # Keep iteration history
            gr.skip(),  # Keep iteration slider
            gr.skip(),  # Keep iteration info
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

    # Wire reconstruction button with all outputs (now includes state)
    reconstruct_btn.click(
        fn=run_reconstruction_ui,
        inputs=[t_slider, z_slider, current_data_xr, current_pixel_scales],
        outputs=[
            comparison_slider,
            loss_plot,
            status_text,
            iteration_history,
            iteration_slider,
            iteration_info,
        ],
    )

    # Wire iteration scrubbing
    iteration_slider.change(
        fn=scrub_iterations,
        inputs=[iteration_slider, iteration_history],
        outputs=[comparison_slider, iteration_info],
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
