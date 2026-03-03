"""Integration test that actually clicks buttons like the user does."""

import pytest


@pytest.fixture
def widget_with_temp_dirs(make_napari_viewer, tmp_path):
    """Create widget with temporary input/output directories."""
    from waveorder.plugin import tab_recon
    from waveorder.plugin.main_widget import MainWidget

    # Reset HAS_INSTANCE to avoid state leakage between tests
    tab_recon.HAS_INSTANCE = {"val": False, "instance": None}

    viewer = make_napari_viewer()
    widget = MainWidget(viewer)

    # Create temp directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Set up the reconstruction tab with directories
    recon_tab = widget.ui.tab_reconstruction
    recon_tab.input_directory = str(input_dir)
    recon_tab.output_directory = str(output_dir)
    recon_tab.data_input_LineEdit.value = str(input_dir)

    yield viewer, widget, recon_tab

    # Cleanup: Reset HAS_INSTANCE after test
    tab_recon.HAS_INSTANCE = {"val": False, "instance": None}


def test_click_new_button_birefringence(widget_with_temp_dirs):
    """Test clicking 'New' button with birefringence checkbox.

    This simulates the exact user workflow:
    1. Check the birefringence checkbox
    2. Click the 'New' button
    3. Verify model was created without errors
    """
    viewer, widget, recon_tab = widget_with_temp_dirs

    # Step 1: Check the birefringence checkbox
    bire_checkbox = recon_tab.modes_selected["birefringence"]["Checkbox"]
    bire_checkbox.value = True

    # Verify checkbox is checked
    assert bire_checkbox.value is True, "Birefringence checkbox should be checked"

    # Step 2: Click the 'New' button by triggering the connected function
    # This is what happens when user clicks the button
    initial_model_count = len(recon_tab.pydantic_classes)

    # Trigger the button's connected function
    recon_tab.build_acq_contols()

    # Step 3: Verify model was created
    assert len(recon_tab.pydantic_classes) > initial_model_count, "A new model should have been created"

    # Verify the model has the correct structure
    model_data = recon_tab.pydantic_classes[-1]  # Get the last (newest) model

    assert "container" in model_data, "Model should have a container"
    assert "selected_modes" in model_data, "Model should track selected modes"
    assert "birefringence" in model_data["selected_modes"], "Birefringence should be in selected modes"

    # Verify the container has birefringence settings
    container = model_data["container"]
    assert hasattr(container, "birefringence"), "Container should have birefringence attribute"


def test_click_new_button_phase(widget_with_temp_dirs):
    """Test clicking 'New' button with phase checkbox."""
    viewer, widget, recon_tab = widget_with_temp_dirs

    # Check the phase checkbox
    phase_checkbox = recon_tab.modes_selected["phase"]["Checkbox"]
    phase_checkbox.value = True

    # Click 'New' button
    initial_model_count = len(recon_tab.pydantic_classes)
    recon_tab.build_acq_contols()

    # Verify model was created
    assert len(recon_tab.pydantic_classes) > initial_model_count, "A new model should have been created"

    model_data = recon_tab.pydantic_classes[-1]
    container = model_data["container"]

    assert hasattr(container, "phase"), "Container should have phase attribute"


def test_click_new_button_birefringence_and_phase(widget_with_temp_dirs):
    """Test clicking 'New' with both birefringence and phase checked."""
    viewer, widget, recon_tab = widget_with_temp_dirs

    # Check both checkboxes
    recon_tab.modes_selected["birefringence"]["Checkbox"].value = True
    recon_tab.modes_selected["phase"]["Checkbox"].value = True

    # Click 'New' button
    initial_model_count = len(recon_tab.pydantic_classes)
    recon_tab.build_acq_contols()

    # Verify model was created
    assert len(recon_tab.pydantic_classes) > initial_model_count

    model_data = recon_tab.pydantic_classes[-1]
    container = model_data["container"]

    # Should have both
    assert hasattr(container, "birefringence"), "Should have birefringence"
    assert hasattr(container, "phase"), "Should have phase"


def test_click_new_button_fluorescence(widget_with_temp_dirs):
    """Test clicking 'New' button with fluorescence checkbox."""
    viewer, widget, recon_tab = widget_with_temp_dirs

    # Check fluorescence
    fluor_checkbox = recon_tab.modes_selected["fluorescence"]["Checkbox"]
    fluor_checkbox.value = True

    # Click 'New' button
    initial_model_count = len(recon_tab.pydantic_classes)
    recon_tab.build_acq_contols()

    # Verify model was created
    assert len(recon_tab.pydantic_classes) > initial_model_count

    model_data = recon_tab.pydantic_classes[-1]
    container = model_data["container"]

    assert hasattr(container, "fluorescence"), "Should have fluorescence"


def test_multiple_models_created(widget_with_temp_dirs):
    """Test creating multiple models by clicking 'New' multiple times."""
    viewer, widget, recon_tab = widget_with_temp_dirs

    # Create first model (birefringence)
    recon_tab.modes_selected["birefringence"]["Checkbox"].value = True
    recon_tab.build_acq_contols()
    assert len(recon_tab.pydantic_classes) == 1

    # Create second model (phase)
    recon_tab.modes_selected["birefringence"]["Checkbox"].value = False
    recon_tab.modes_selected["phase"]["Checkbox"].value = True
    recon_tab.build_acq_contols()
    assert len(recon_tab.pydantic_classes) == 2

    # Verify both models exist
    assert "birefringence" in recon_tab.pydantic_classes[0]["selected_modes"]
    assert "phase" in recon_tab.pydantic_classes[1]["selected_modes"]
