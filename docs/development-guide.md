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

4. Optionally, for the co-development of [`waveorder`](https://github.com/mehta-lab/waveorder) and `recOrder`:

    > Note that `pip` will raise an 'error' complaining that the dependency of `recOrder-napari` has been broken if you do the following.
    > This does not affect the installation, but can be suppressed by removing [this line](https://github.com/mehta-lab/recOrder/blob/5bc9314a9bacf6f4e235eaffb06c297cf20e4b65/setup.cfg#L40) before installing `waveorder`.
    > (Just remember to revert the change afterwards!)
    > We expect a nicer behavior to be possible once we release a stable version of `waveorder`.

    ```sh
    cd # where you want to clone the repo
    git clone https://github.com/mehta-lab/waveorder.git
    pip install ./waveorder -e ".[dev]"
    ```

    > Importing from a local and/or editable package can cause issues with static code analyzers such due to the absence of the source code from the paths they expect.
    > VS Code users can refer to [this guide](https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#common-questions-and-issues) to resolve typing warnings.

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

Open `./recOrder/plugin/gui.ui` in `QT Creator` and make your changes.

Next, convert the `.ui` to a `.py` file with:

```sh
pyuic5 -x gui.ui -o gui.py
```

Note: `pyuic5` is installed alongside `PyQt5`, so you can expect to find it installed in your `recOrder` conda environement.

Finally, change the `gui.py` file's to import `qtpy` instead of `PyQt5` to adhere to [napari plugin best practices](https://napari.org/stable/plugins/best_practices.html#don-t-include-pyside2-or-pyqt5-in-your-plugin-s-dependencies).
On macOS, you can modify the file in place with:

```sh
sed -i '' 's/from PyQt5/from qtpy/g' gui.py
```

> This is specific for BSD `sed`, omit `''` with GNU.

Note: although much of the GUI is specified in the generated `recOrder_ui.py` file, the `main_widget.py` file makes extensive modifications to the GUI.

## Make `git blame` ignore formatting commits

**Note:** `git --version` must be `>=2.23` to use this feature.

If you would like `git blame` to ignore formatting commits, run this line:

```sh
 git config --global blame.ignoreRevsFile .git-blame-ignore-revs
```

The `\.git-blame-ignore-revs` file contains a list of commit hashes corresponding to formatting commits.
If you make a formatting commit, please add the commit's hash to this file.
