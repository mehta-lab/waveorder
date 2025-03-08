# `waveorder` CLI examples

`waveorder` uses a configuration-file-based command-line inferface (CLI) to
calculate transfer functions and apply these transfer functions to datasets.

This page demonstrates `waveorder`'s CLI.

## Getting started

### 1. Check your installation
First, [install `waveorder`](../docs/software-installation-guide.md) and run
```bash
waveorder
```
in a shell. If `waveorder` is installed correctly, you will see a usage string and
```
waveorder: Computational Toolkit for Label-Free Imaging
```

### 2. Download and convert a test dataset
Next, [download the test data from zenodo (47 MB)](https://zenodo.org/record/6983916/files/recOrder_test_data.zip?download=1), and convert a dataset to the latest version of `.zarr` with
```
cd /path/to/
iohub convert -i /path/to/test_data/2022_08_04_recOrder_pytest_20x_04NA/2T_3P_16Z_128Y_256X_Kazansky_1/
-o ./dataset.zarr
```

You can view the test dataset with
```
napari ./dataset.zarr --plugin waveorder
```

### 3. Run a reconstruction
Run an example reconstruction with
```
waveorder reconstruct ./dataset.zarr/0/0/0 -c /path/to/waveorder/examples/settings/birefringence-and-phase.yml -o ./reconstruction.zarr
```
then view the reconstruction with
```
napari ./reconstruction.zarr --plugin waveorder
```

Try modifying the configuration file to see how the regularization parameter changes the results.

## FAQ
1. **Q: Which configuration file should I use?**

    If you are acquiring:

    **3D data with calibrated liquid-crystal polarizers via `waveorder`** use `birefringence.yml`.

    **3D fluorescence data** use `fluorescence.yml`.

    **3D brightfield data** use `phase.yml`.

    **Multi-modal data**, start by reconstructing the individual modaliities, each with a single config file and CLI call. Then combine the reconstructions by ***TODO: @Ziwen do can you help me append to the zarrs to help me fix this? ***

2. **Q: Should I use `reconstruction_dimension` = 2 or 3?

    If your downstream processing requires 3D information or if you're unsure, then you should use `reconstruction_dimension = 3`. If your sample is very thin compared to the depth of field of the microscope, if you're in a noise-limited regime, or if your downstream processing requires 2D information, then you should use `reconstruction_dimension = 2`. Empirically, we have found that 2D reconstructions reduce the noise in our reconstructions because it uses 3D information to make a single  estimate for each pixel.

3. **Q: What regularization parameter should I use?**

    We recommend starting with the defaults then testing over a few orders of magnitude and choosing a result that isn't too noisy or too smooth.

### Developers note

These configuration files are automatically generated when the tests run. See `/tests/cli_tests/test_settings.py` - `test_generate_example_settings`.

To keep these settings up to date, run `pytest` locally when `cli/settings.py` changes.
