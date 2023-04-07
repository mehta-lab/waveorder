# `recOrder` example scripts

These scripts demonstrate four types of reconstruction:

1. 3D brightfield data to 3D phase in `3D-bf-to-3D-phase.py`. 
2. 3D brightfield data to 2D phase in `3D-bf-to-2D-phase.py`.
3. 3D + polarization data to retardance, orientation, and 3D phase in `3D-pol-to-birefringence.py`.
4. 3D fluorescence data to 3D fluorophore density (fluorescence deconvolution) in `3D-fluor-to-3D-density.py`.    

## Getting started

### 1. Check your installation
First, [install `recOrder`](../docs/software-installation-guide.md) and run any of these scripts with 
```bash
python <script-name.py>
```
Running these scripts without modification will test your installation by running a reconstruction on a simulated phantom. A successful run will open a `napari` window with simulated data and a reconstruction. 

### 2. Reconstruct test data
Next, [download the test data from zenodo (47 MB)](https://zenodo.org/record/6983916/files/recOrder_test_data.zip?download=1), and modify the script to load the test data. For example, in `3D-bf-to-2D-phase.py`, replace `/path/to/ome-tiffs/or/zarr/store/` with `/path/to/recOrder_test_data/2022_08_04_recOrder_pytest_20x_04NA_BF/2T_3P_16Z_128Y_256X_Kazansky_BF_1`.

### 3. Load and reconstruct your data

Start by loading your data (our readers currently support `.tiff`, `ome.tiff`, `NDTiff`, and `.zarr`, but any `numpy` array will work). Fill in your imaging parameters (or connect your metadata to these parameters), and prototype with this modified script. 

We recommend prototyping a reconstruction with a single position and time point so that you can perform initial iterations quickly. 

### 4. Sweep your reconstruction parameters

You may need to test several parameters to find a value that yields the best results for your application. For example, choosing a regularization parameter is commonly semi-empirical: we recommend choosing a regularization parameter that gives results that aren't too noisy or too smooth. 

The `sweep-regularization.py` script demonstrates a `3D-bf-to-2D-phase` reconstruction with multiple regularization parameters. We recommend running and understanding this script before modifying your reconstruction script to sweep over regularization or other uncertain parameters to help you settle on a set of reconstruction parameters. 

### 5. Reconstruct a multi-modal dataset in a single script

You may need to perform several reconstructions on a multi-modal dataset. For example, you would like to perform a `3D-fluor-to-3D-density` reconstruction on the fluorescence channels and a `3D-bf-2D-phase` reconstruction on the brightfield channel. 

The `multi-modal-recon.py` script demonstrates this type of reconstruction. We recommend running and understanding this script before modifying single-modality reconstructions to run a multi-modal reconstruction. 

### 6. Parallelize over positions or time points

Once you've settled on a script that performs a reconstruction, the script can be applied to multiple datasets with a python `for` loop (slowest), `multiprocessing` (faster, see `parallel-reconstruct.py` for an example), or batch processing with an HPC scheduler e.g. `slurm` (fastest). 

## FAQ
1. **Q: Which script should I use?**

    If you are acquiring:

    **3D data with calibrated liquid-crystal polarizers via `recOrder`** use `3D-pol-to-birefringence.py`.

    **3D fluorescence data** use `3D-fluor-to-3D-density.py`.

    **3D brightfield data** use `3D-bf-to-3D-phase.py` or `3D-bf-to-2D-phase.py`, and decide if you need a 3D or 2D phase reconstruction. 

    If your downstream processing requires 3D information or if you're unsure, then you should use `3D-bf-to-3D-phase.py`. If your sample is very thin compared to the depth of field of the microscope, if you're in a noise-limited regime, or if your downstream processing requires 2D phase information, then you should use `3D-bf-to-2D-phase.py`. Empirically, we have found that `3D-bf-to-2D-phase.py` reduces the noise in our reconstructions because it uses 3D information to make a single phase estimate for each pixel. 

    **Multi-modal data**, start by reconstructing the individual modaliities, then combine the reconstructions using `multi-modal-recon.py` as a guide.  

2. **Q: What regularization parameter should I use?**

    We recommend starting with the defaults then testing over a few orders of magnitude and choosing a result that isn't too noisy or too smooth.
