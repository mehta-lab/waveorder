# recOrder
[![License](https://img.shields.io/pypi/l/recOrder-napari.svg)](https://github.com/recOrder/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/recOrder-napari)
[![Downloads](https://pepy.tech/badge/recOrder-napari)](https://pepy.tech/project/recOrder-napari)
[![Python package index](https://img.shields.io/pypi/v/recOrder-napari.svg)](https://pypi.org/project/recOrder-napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This package offers a comprehensive pipeline, command line interface, and napari plugin for quantitative label-free microscopy.

In this repo you will find python tools and UI that allow the user to calibrate microscope hardware, acquire multi-modal data, reconstruct density and anisotropy, and visualize the data.

The acquisition, calibration, background correction, reconstruction, and applications of QLIPP are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

``` Syuan-Ming Guo, Li-Hao Yeh, Jenny Folkesson, Ivan E Ivanov, Anitha P Krishnan, Matthew G Keefe, Ezzat Hashemi, David Shin, Bryant B Chhun, Nathan H Cho, Manuel D Leonetti, May H Han, Tomasz J Nowakowski, Shalin B Mehta, "Revealing architectural order with quantitative label-free imaging and deep learning," eLife 2020;9:e55502 DOI: 10.7554/eLife.55502 (2020).```

recOrder is to be used alongside the QLIPP module, whose design has been optimized to fit on a conventional widefield microscope (Panel A below).  The QLIPP module allows for the collection of label-free information consisting of the intrinsic anisotropy of the sample and its relative phase (density).  All of these measurements are collected through compensated, polarization diverse illumination and quantitatively recovered through recOrder's computational reconstruction pipeline.  The overall structure of recOrder can be visualized below in Panel B, highlighting the two different usage modes and their features: graphical user interface (GUI) through napari and command line interfact (CLI).

![Flow Chart](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_Fig1_Overview.png?raw=true)

## Dataset

[Slides](https://doi.org/10.5281/zenodo.5135889) and a [dataset](https://doi.org/10.5281/zenodo.5178487) shared during a workshop on QLIPP and recOrder can be found on Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)

## Installation

**Easy installation:**

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment  
```
conda create -n recorder python
conda activate recorder
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
conda create -n recorder python
conda activate recorder
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

## License

Chan Zuckerberg Biohub Software License

This software license is the 2-clause BSD license plus clause a third clause
that prohibits redistribution and use for commercial purposes without further
permission.

Copyright © 2019. Chan Zuckerberg Biohub.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3.	Redistributions and use for commercial purposes are not permitted without
the Chan Zuckerberg Biohub's written permission. For purposes of this license,
commercial purposes are the incorporation of the Chan Zuckerberg Biohub's
software into anything for which you will charge fees or other compensation or
use of the software to perform a commercial service for a third party.
Contact ip@czbiohub.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
