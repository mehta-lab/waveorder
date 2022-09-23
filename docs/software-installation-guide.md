# Software Installation Guide

1. (Optional but recommended) install [`conda`](https://github.com/conda-forge/miniforge) and create a virtual environment  

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

2. Install `recOrder-napari`:

    ```sh
    pip install recOrder-napari
    ```

3. To use the GUI: open `napari` with `recOrder-napari`:

    ```sh
    napari -w recOrder-napari
    ```

4. View command-line help by running

    ```sh
    recOrder.help
    ```

5. To acquire data via `Micromanager`, follow the [microscope installation guide](./microscope-installation-guide.md).

## GPU acceleration (Optional)

`recOrder` supports NVIDIA GPU computation with the `cupy` package. Follow [these instructions](https://github.com/cupy/cupy) to install `cupy` and check its installation with ```import cupy```.
To enable gpu processing, set ```use_gpu: True``` in the config files.
