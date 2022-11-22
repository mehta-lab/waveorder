# `recOrder` example scripts

These scripts demonstrate three types of reconstruction:

1. 3D brightfield data to 3D phase in `3D-bf-to-3D-phase.py`. 
2. 3D brightfield data to 2D phase in `3D-bf-to-2D-phase.py`.
3. 4D polarization data to retardance, orientation, and 3D phase in `recon-qlipp.py`.   

## Getting started
First, [install `recOrder`](../docs/software-installation-guide.md) and run these scripts with 
```bash
python <script-name.py>
```
Running any of these scripts without modification will test your installation by running a reconstruction on random data. A successful run will open a `napari` window with results reconstructed from random data. 

Next, [download the test data from zenodo (47 MB)](https://zenodo.org/record/6983916/files/recOrder_test_data.zip?download=1), and modify the script to load the test data. For example, in `3D-bf-to-2D-phase.py`, replace `/path/to/ome-tiffs/or/zarr/store/` with `/path/to/recOrder_test_data/2022_08_04_recOrder_pytest_20x_04NA_BF/2T_3P_16Z_128Y_256X_Kazansky_BF_1`.

Finally, modify the script to process your data. Start by loading your data (our readers currently support `.tiff`, `ome.tiff`, `NDTiff`, and `.zarr`, but any `numpy` array will work). Fill in your imaging parameters (or connect your metadata to these parameters), and prototype with a script. 

We recommend prototyping a reconstruction with a single position and time point. You may need to test several regularization parameters to find a value that yields results that aren't too noisy or too smooth. 

After prototyping, the script can be applied to multiple datasets with a python `for` loop (slowest), `multiprocessing` (faster), or batch processing with an HPC scheduler e.g. `slurm` (fastest). 

## FAQ
Q: Which script should I use?

A: If you are reconstructed data acquired with calibrated liquid-crystal polarizers, use `recon-qlipp.py`. Otherwise, you will need to decide between a 3D or 2D phase reconstruction. 

If your downstream processing requires 3D information or if you're unsure, then you should use `3D-bf-to-3D-phase.py`. If your sample is very thin compared to the depth of field of the microscope, if you're in a noise-limited regime, or if your downstream processing requires 2D phase information, then you should use `3D-bf-to-2D-phase.py`. Empirically, we have found that `3D-bf-to-2D-phase.py` reduces the noise in our reconstructions because it uses 3D information to make a single phase estimate for each pixel. 

Q: What regularization parameter should I use?

We recommend starting with the defaults then testing over a few orders of magnitude and choosing a result that isn't too noisy or too smooth.
