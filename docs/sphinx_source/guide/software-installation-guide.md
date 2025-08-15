# Software Installation Guide

1. (Optional but recommended) install [`conda`](https://github.com/conda-forge/miniforge) and create a virtual environment

    ```sh
    conda create -y -n waveorder python=3.12
    conda activate waveorder
    ```

2. Install `waveorder`:

    ```sh
    pip install waveorder
    ```

3. To use the GUI: open `napari` with `waveorder`:

    ```sh
    napari -w waveorder
    ```

4. View command-line help by running

    ```sh
    waveorder
    ```

5. To acquire data via Micro-Manager`, follow the [microscope installation guide](./microscope-installation-guide.md).
