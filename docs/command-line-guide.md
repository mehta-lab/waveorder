# Command Line Guide

This command-line interface command allows the user to move through quantitative reconstruction of phase and birefringence through a user-defined config file.  The user should copy one of the example config files from `/examples/example_configs/`.

## Using the command

The main command for reconstruction is `recOrder.reconstruct` with the following flags:

```
recOrder.reconstruct

--config (str) [REQUIRED] path to the config.yml file created by the user.  All of the commands below can be specified within the config file.  

Any of these optional commands will override the corresponding fields that the user may have defined or omitted in the config file.

--method (str) [optional with config] Method of reconstruction: PhaseFromBF, QLIPP
--mode (str) [optional with config] Mode of reconstruction: 2D or 3D
--data_dir (str) [optional with config] path to raw data folder (root zarr path or folder containing micromanager tif files)
--save_dir (str) [optional with config] path to the directory in which the processed data will be saved
--name (str) [optional] name of the zarr store root folder to be saved in the save_dir.  If not specified, it will use the name of the raw data.
--overwrite (bool) True/False: whether or not to overwrite any existing processed zarr data store under the <save_dir>/<name>.zarr 
```

At any time you can type the `recOrder.help` command in order to display the available CLI commands:

![Screen Shot 2022-02-15 at 10 05 07 AM](https://user-images.githubusercontent.com/56048121/154122180-b6509882-d404-4f42-92c4-b66124a57cec.png)


## Config Specification

### Dataset Config Specification
```
dataset:
  method: 'QLIPP'
  # Reconstruction Method 'QLIPP', or 'PhaseFromBF'.

  mode: '2D'
  # Mode for reconstruction, '2D' or '3D'

  data_dir: '/path/to/raw/data'
  # (str) path to the raw data directory (folder which holds .tif files) or the root .zarr file

  save_dir: '/path/to/save/directory'
  # (str) path to the directory in which the data will be saved

  data_type [optional]: 'ometiff'
  # (str) the datatype of the raw data. One of 'ometiff', 'singlepagetiff', or 'zarr'. If not specified the data reader will infer the 
  datatype

  data_save_name: 'Test_Data'
  # (str) Name of the zarr dataset that will be written.

  positions: [1]
  # (str or list) Positions within the dataset that are to be analyzed.
  #   'all'
  #   !!python/tuple [0, N] for pos 0 through N
  #   [0, 1, 5] for pos 0, 1, and 5
  #   [!!python/tuple [0, 15], [19, 25]] for pos 0 through 15, 19, and 25

  timepoints: [0]
  # (str or list) timepoints within the dataset that are to be analyzed.
  #   'all'
  #   !!python/tuple [0, N] for pos 0 through N
  #   [0, 1, 5] for pos 0, 1, and 5
  #   [!!python/tuple [0, 15], [19, 25]] for timepoints 0 through 15, 19, and 25

  background [optional, required if background_correction is not 'None']: './pytest_temp/pipeline_test/2021_06_11_recOrder_pytest_20x_04NA/BG'
  # (str) Background folder within the experiment folder

  calibration_metadata [required for QLIPP pipeline]: './pytest_temp/pipeline_test/2021_06_11_recOrder_pytest_20x_04NA/calib_metadata.txt'
  # (str) path to the qlipp calibration metadata file
```

### Processing

This is the section that allows the user to define the output of the reconstruction.

```
processing:
  output_channels: ['Retardance', 'Orientation', 'Brightfield', 'Phase2D', 'S0', 'S1', 'S2', 'S3']

  # (list) Any combination of the following values.
  #    'Retardance', 'Orientation','Brightfield', 'Phase3D'. 'S0', 'S1', 'S2', 'S3'
  #     order of the channels specifies the order in which they will be written

  background_correction: 'local_fit'
  # (str) Background correction method, one of the following
  ##   'None': no background correction will be performed
  ##   'local': for estimating background with scipy uniform filter
  ##   'local_fit': for estimating background with polynomial fit
  ##   'global': for normal background subtraction with the provided background

  use_gpu: False
  # (bool) Option to use GPU processing if True (require cupy to be installed)

  gpu_id: 0
  # (int) ID of GPU to be used

  ########################################
  #    PHASE RECONSTRUCTION PARAMETERS   #
  ########################################

  wavelength: 532
  # (int) wavelength of the illumination in nm

  pixel_size: 6.5
  # (float) Camera pixel size in the unit of um

  magnification: 20
  # (float) Magnification of the objective

  NA_objective: 0.55
  # (float) Numerical aperture of the objective

  NA_condenser: 0.4
  # (float) Numerical aperture of the condenser

  n_objective_media: 1.0
  # (float) Refractive index of the objective immersion oil

  focus_zidx: 40
  # (int) Index of the focused z slice of the dataset for 2D phase reconstruction

  pad_z: 0
  # (int) Number of z slices padded above and below the dataset for 3D phase reconstruction to avoid boundary artifacts

  ## Denoiser parameters ##
  phase_denoiser_2D: 'Tikhonov'
    # (str) Options of denoiser for 2D phase reconstruction
    ##   'Tikhonov' or 'TV' (total variation)

  #### 2D Tikhonov parameter ####
  # if noise is higher raise the regularization parameter an order of magnitude to see if the recon is better

  Tik_reg_abs_2D: 1.0e-4
  # (float) Tikhonov regularization parameter for 2D absorption
  ##   1.0e-3 should work generally when noise is low

  Tik_reg_ph_2D: 1.0e-4
  # (float) Tikhonov regularization parameter for 2D phase
  ##   1.0e-3 should work generally when noise is low

  rho_2D: 1
  # (float) rho parameters in the 2D ADMM formulation
  ##   1 is generally good, no need to tune

  itr_2D: 50
  # (int) Number of iterations for 2D TV denoiser
  ##   50 is generally good, no need to tune

  TV_reg_abs_2D: 1.0e-4
  # (float) TV regularization parameter for 2D absorption
  ##   1e-4 is generally good

  TV_reg_ph_2D: 1.0e-4
  # (float) TV regularization parameter for 2D phase
  ##   1e-4 is generally good

  # -------- 3D ---------

  phase_denoiser_3D: 'Tikhonov'
  # (str) Options of denoiser for 3D phase reconstruction
  ##   'Tikhonov' or 'TV' (total variation)

  #### 3D Tikhonov parameters ####
  # if noise is higher raise an order of magnitude to see if the recon is better

  Tik_reg_ph_3D: 1.0e-4
  # (float) Tikhonov regularization parameter for 3D phase
  ##   1.0e-3 to 1.0e-4 should work generally when noise is low

  #### 3D TV parameters ####
  # For more noisy data, raise TV_reg to enforce stronger denoising effect

  rho_3D: 1.0e-3
  # (float) rho parameters in the 2D ADMM formulation
  ##   1.0e-3 is generally good, no need to tune

  itr_3D: 50
  # (int) Number of iterations for 3D TV denoiser
  ##   50 is generally good, no need to tune

  TV_reg_ph_3D: 5.0e-5
  # (float) TV regularization parameter for 3D phase
  ##   5.0e-5 is generally good
```

# recOrder Data Conversion

This section will allow the user to convert micromanager ome-tif data to ome-zarr data.  OME-zarr format is useful for compression storage and increases the efficiency of reconstruction with access to lazy loading of the data.

## Using the command

The main command for reconstruction is `recOrder.convert` with the following flags:

```
recOrder.convert

--input [required] (str) path to folder containing micromanager tif files
--output [required] (str) full path to save the ome-zarr data, i.e. /path/to/Data.zarr
--data_type [optional] (str) micromananger data-type: ometiff, singlepagetiff
--replace_pos_names [optional] (bool) True/False whether to replace zarr position names with ones in the user-defined position list in micro-manager metadata. Default is False
--format_hcs [optional] (bool) if a tiled micromanager dataset, format in ome-zarr HCS format (creates a tiled layout of the positions from metadata). Default is False.

```






