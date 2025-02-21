# Software Installation Guide

1. (Optional but recommended) install [`conda`](https://github.com/conda-forge/miniforge) and create a virtual environment  

    ```sh
    conda create -y -n recOrder python=3.10
    conda activate recOrder
    ```

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
    recOrder
    ```

5. To acquire data via Micro-Manager`, follow the [microscope installation guide](./microscope-installation-guide.md).
