# recOrder
[![License](https://img.shields.io/pypi/l/recOrder-napari.svg)](https://github.com/mehta-lab/recOrder/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/recOrder-napari)
[![Downloads](https://pepy.tech/badge/recOrder-napari)](https://pepy.tech/project/recOrder-napari)
[![Python package index](https://img.shields.io/pypi/v/recOrder-napari.svg)](https://pypi.org/project/recOrder-napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This package offers a comprehensive pipeline, command line interface, and napari plugin for quantitative label-free microscopy.

In this repo you will find python tools and UI that allow the user to calibrate microscope hardware, acquire multi-modal data, reconstruct density and anisotropy, and visualize the data.

The acquisition, calibration, background correction, reconstruction, and applications of QLIPP are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

``` Syuan-Ming Guo, Li-Hao Yeh, Jenny Folkesson, Ivan E Ivanov, Anitha P Krishnan, Matthew G Keefe, Ezzat Hashemi, David Shin, Bryant B Chhun, Nathan H Cho, Manuel D Leonetti, May H Han, Tomasz J Nowakowski, Shalin B Mehta, "Revealing architectural order with quantitative label-free imaging and deep learning," eLife 2020;9:e55502 DOI: 10.7554/eLife.55502 (2020).```

recOrder is to be used alongside the QLIPP module, whose design has been optimized to fit on a conventional widefield microscope (Panel A below).  The QLIPP module allows for the collection of label-free information consisting of the intrinsic anisotropy of the sample and its relative phase (density).  All of these measurements are collected through compensated, polarization diverse illumination and quantitatively recovered through recOrder's computational reconstruction pipeline.  The overall structure of recOrder is shown in Panel B, highlighting the two different usage modes and their features: graphical user interface (GUI) through napari and command line interfact (CLI).

![Flow Chart](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_Fig1_Overview.png?raw=true)

## Dataset

[Slides](https://doi.org/10.5281/zenodo.5135889) and a [dataset](https://doi.org/10.5281/zenodo.5178487) shared during a workshop on QLIPP and recOrder can be found on Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)

## Installation

**Easy installation:**

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment  
```
conda create -n recOrder python
conda activate recOrder
```
Install napari:
```
pip install "napari[all]"
```
Open `napari` and use the `Plugin > Install/Uninstall Plugins...` menu to install `recOrder-napari`.

**Developer installation:**

Install [`git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and `conda` (either [anaconda](https://www.anaconda.com/products/distribution) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Create a conda environment dedicated to `recOrder`:
```
conda create -n recOrder python
conda activate recOrder
```

Clone this repository:
```buildoutcfg
git clone https://github.com/mehta-lab/recOrder.git
```

Install recOrder and its dependencies:
```buildoutcfg
cd recOrder
pip install -e ".[dev]"
```

**Optional `napari` plugin**: `recOrder` includes a `napari` plugin that can be used to acquire data via MicroManager. To install `napari` use
```
pip install "napari[all]"
```
To run the `recOrder` plugin use
```
napari -w recOrder-napari
```

To acquire data via MicroManager, follow the [instructions on the wiki](https://github.com/mehta-lab/recOrder/wiki/recOrder-Installation-and-MicroManager-Setup-Guide).

**Optional GPU**: `recOrder` supports NVIDIA GPU computation with the `cupy` package. Follow [these instructions](https://github.com/cupy/cupy) to install `cupy` and check its installation with ```import cupy```. To enable gpu processing, set ```use_gpu: True``` in the config files.

## Command-line usage
Type `recOrder.help` for instructions on the two command-line modes: `recOrder.reconstruct` and `recOrder.convert`.

### `recOrder.reconstruct`

`recOrder.reconstruct` uses configuration files to select reconstruction parameters. Start with an example configuration file `/examples/example_configs/config_example_qlipp.yml` and modify the parameters to match your dataset.

Run the reconstruction with
```buildoutcfg
recOrder.reconstruct --config <path/to/config.yml>
```

The following command-line arguments override parameters specified in the configuration file:

   ```
   --method (str) method of reconstruction: QLIPP,IPS,UPTI'
   --mode (str) mode of reconstruction: 2D, 3D'
   --data_dir (str) path to raw data folder'
   --save_dir (str) path to folder where reconstructed data will be saved'
   --name (str) name under which to save the reconstructed data'
   --config (str) path to configuration file (see /examples/example_configs')
   --overwrite (bool) True/False whether or not to overwrite data that exists under save_dir/name'
   ```

For example, this command uses the `QLIPP` reconstruction method even if the configuration file specifies a different reconstruction method
```buildoutcfg
recOrder.reconstruct --config /path/to/config.yml --method QLIPP
```

### `recOrder.convert`

`recOrder.convert` converts MicroManager `tif` files to `ome-zarr` files. For example

```buildoutcfg
recOrder.convert --input <path/to/mm/tifs> --output <path/to/output.zarr>  --data_type ometiff
```

