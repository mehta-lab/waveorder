# Data schema

This document defines the standard for data acquired with `recOrder`.

## Raw directory organization

Currently, we structure raw data in the following hierarchy:

```text
working_directory/                      # commonly YYYY_MM_DD_exp_name, but not enforced
├── polarization_calibration_0.txt        
│   ...
├── polarization_calibration_<i>.txt    # i calibration repeats
│
├── bg_0
│   ...
├── bg_<j>                              # j background repeats
│   ├── background.zarr             
│   ├── polarization_calibration.txt    # copied into each bg folder 
│   ├── reconstruction.zarr
│   ├── reconstruction_settings.yml     # for use with `recorder reconstruct`
│   └── transfer_function.zarr          # for use with `recorder apply-inv-tf`
│
├── <acq_name_0>_snap_0   
├── <acq_name_0>_snap_1 
│   ├── raw_data.zarr
│   ├── reconstruction.zarr
│   ├── reconstruction_settings.yml
│   └── transfer_function.zarr
│   ...
├── <acq_name_0>_snap_<k>               # k repeats with the first acquisition name
│   ├── raw_data.zarr          
│   ├── reconstruction.zarr
│   ├── reconstruction_settings.yml
│   └── transfer_function.zarr
│   ...
│
├── <acq_name_l>_snap_0                 # l different acquisition names
│   ...
├── <acq_name_l>_snap_<m>               # m repeats for this acquisition name
    ├── raw_data.zarr 
    ├── reconstruction.zarr
    ├── reconstruction_settings.yml
    └── transfer_function.zarr
```

Each `.zarr` contains an [OME-NGFF v0.4](https://ngff.openmicroscopy.org/0.4/) in HCS format with a single field of view. 