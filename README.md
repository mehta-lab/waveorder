# recOrder
[![License](https://img.shields.io/pypi/l/recOrder-napari.svg)](https://github.com/mehta-lab/recOrder/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/recOrder-napari)
[![Downloads](https://pepy.tech/badge/recOrder-napari)](https://pepy.tech/project/recOrder-napari)
[![Python package index](https://img.shields.io/pypi/v/recOrder-napari.svg)](https://pypi.org/project/recOrder-napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This package provides a napari plugin and a command line interface for quantitative label-free microscopy.

In this repository you will find python tools and a napari plugin that allow the user to calibrate microscope hardware, acquire multi-modal data, reconstruct density and anisotropy, and visualize the results.

The acquisition, calibration, background correction, reconstruction, and applications of QLIPP (quantitative label-free imaging with phase and polarization)  are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

``` Syuan-Ming Guo, Li-Hao Yeh, Jenny Folkesson, Ivan E Ivanov, Anitha P Krishnan, Matthew G Keefe, Ezzat Hashemi, David Shin, Bryant B Chhun, Nathan H Cho, Manuel D Leonetti, May H Han, Tomasz J Nowakowski, Shalin B Mehta, "Revealing architectural order with quantitative label-free imaging and deep learning," eLife 2020;9:e55502 DOI: 10.7554/eLife.55502 (2020).```

`recOrder` is to be used alongside a conventional widefield microscope fitted with a universal polarizer (Panel A below).  The universal polarizer allows for the collection of label-free information including the intrinsic anisotropy of the sample and its relative phase (density). These measurements are collected by acquiring data under calibrated, polarization-diverse illumination followed by a computational reconstruction.  The overall structure of `recOrder` is shown in Panel B, highlighting the two different usage modes and their features: graphical user interface (GUI) through a napari plugin, and a command line interface (CLI).

![Flow Chart](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_Fig1_Overview.png?raw=true)

## Dataset

[Slides](https://doi.org/10.5281/zenodo.5135889) and a [dataset](https://doi.org/10.5281/zenodo.5178487) shared during a workshop on QLIPP and recOrder can be found on Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)

## Quick Start

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment  
```
conda create -y -n recOrder python=3.9
conda activate recOrder
```
Install `napari` and `recOrder-napari`:
```
pip install "napari[all]" recOrder-napari
```
Open `napari` with `recOrder-napari`:
```
napari -w recOrder-napari
```
View command-line help by running
```
recOrder.help
```

For more help, see [`recOrder`s documentation](./docs). 

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
