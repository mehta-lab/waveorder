# `recOrder` development guide

## Install `recOrder` for development

1. Install [conda](https://github.com/conda-forge/miniforge) and create a virtual environment:

    ```sh
    conda create -y -n recOrder python=3.9
    conda activate recOrder
    ```

    > *Apple Silicon users please use*:
    >
    > ```sh
    > CONDA_SUBDIR=osx-64 conda create -y -n recOrder python=3.9
    > conda activate recOrder
    > ```
    >
    > Reason: `napari` requires `PyQt5` which is not available for `arm64` from PyPI wheels.
    > Specifying `CONDA_SUBDIR=osx-64` will install an `x86_64` version of `python` which has `PyQt5` wheels available.

2. Clone the `recOrder` directory:

    ```sh
    git clone https://github.com/mehta-lab/recOrder.git
    ```

3. Install `recOrder` in editable mode with development dependencies

    ```sh
    cd recOrder
    pip install -e ".[dev]"
    ```

## Set up a development environment

### Code linting

We are not currently specifying a code linter as most modern Python code editors already have their own. If not, add a plugin to your editor to help catch bugs pre-commit!

### Code formatting

We use `black` to format Python code, and a specific version is installed as a development dependency. Use the `black` in the `recOrder` virtual environment, either from commandline or the editor of your choice.

> *VS Code users*: Install the [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) plugin. Press `^/⌘ ⇧ P` and type 'format document with...', choose the Black Formatter and start formatting!

### Docstring style

The [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings are used in `recOrder`.

> *VS Code users*: [this popular plugin](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) helps auto-generate most popular docstring styles (including `numpydoc`).

## Run automated tests

From within the `recOrder` directory run:

```sh
pytest
```

Running `pytest` for the first time will download ~50 MB of test data from Zenodo, and subsequent runs will reuse the downloaded data.

## Run manual tests

Although many of `recOrder`'s tests are automated, many features require manual testing. The following is a summary of features that need to be tested manually before release:

* Install a compatible version of micromanager and check that `recOrder` can connect.
* Perform calibrations with and without an ROI; with and without a shutter configured in micromanager, in 4- and 5-state modes; and in MM-Voltage, MM-Retardance, and DAC modes (if the TriggerScope is available).  
* Test "Load Calibration" and "Calculate Extinction" buttons.
* Test "Capture Background" button.
* Test the "Acquire Birefringence" button on a background FOV. Does a background-corrected background acquisition give random orientations?
* Test the four "Acquire" buttons with varied combinations of 2D/3D, background correction settings, "Phase from BF" checkbox, and regularization parameters.
* Use the data you collected to test "Offline" mode reconstructions with varied combinations of parameters.  

## GUI development

We use `QT Creator` for large parts of `recOrder`'s GUI. To modify the GUI, install `QT Creator` from [its website](https://www.qt.io/product/development-tools) or with `brew install --cask qt-creator`

Open `/recOrder/recOrder/plugin/qtdesigner/recOrder_ui.ui` in `QT Creator` and make your changes. 

Finally, convert the `.ui` to a `.py` file with 
```sh
pyuic5 -x recOrder_ui.ui -o recOrder_ui.py
```
Note: `pyuic5` is installed alongside `PyQt5`, so you can expect to find it installed in your `recOrder` conda environement. 

Note: although much of the GUI is specified in the generated `recOrder_ui.py` file, the `main_widget.py` file makes extensive modifications to the GUI. 