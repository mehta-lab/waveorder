# recOrder
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/recOrder-napari)
[![Python package index download statistics](https://img.shields.io/pypi/dm/recOrder-napari.svg)](https://pypistats.org/packages/recOrder-napari)
[![Python package index](https://img.shields.io/pypi/v/recOrder-napari.svg)](https://pypi.org/project/recOrder-napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

`recOrder` is a collection of computational imaging methods. It currently provides QLIPP (quantitative label-free imaging with phase and polarization), phase from defocus, and fluorescence deconvolution. 

[![Unveiling the invisible](https://github.com/mehta-lab/recOrder/blob/main/docs/images/comms_video_screenshot.png?raw=true)](https://www.youtube.com/watch?v=JEZAaPeZhck)

Acquisition, calibration, background correction, reconstruction, and applications of QLIPP are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

```bibtex
Syuan-Ming Guo, Li-Hao Yeh, Jenny Folkesson, Ivan E Ivanov, Anitha P Krishnan, Matthew G Keefe, Ezzat Hashemi, David Shin, Bryant B Chhun, Nathan H Cho, Manuel D Leonetti, May H Han, Tomasz J Nowakowski, Shalin B Mehta, "Revealing architectural order with quantitative label-free imaging and deep learning," eLife 2020;9:e55502 DOI: 10.7554/eLife.55502 (2020).
```

These are the kinds of data you can acquire with `recOrder` and QLIPP:

https://user-images.githubusercontent.com/9554101/271128301-cc71da57-df6f-401b-a955-796750a96d88.mov

https://user-images.githubusercontent.com/9554101/271128510-aa2180af-607f-4c0c-912c-c18dc4f29432.mp4

## What do I need to use `recOrder`
`recOrder` is to be used alongside a conventional widefield microscope. For QLIPP, the microscope must be fitted with an analyzer and a universal polarizer: 

https://user-images.githubusercontent.com/9554101/273073475-70afb05a-1eb7-4019-9c42-af3e07bef723.mp4

For phase-from-defocus or fluorescence deconvolution methods, the universal polarizer is optional.

The overall structure of `recOrder` is shown in Panel B, highlighting the structure of the graphical user interface (GUI) through a napari plugin and the command-line interface (CLI) that allows users to perform reconstructions.

![Flow Chart](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_Fig1_Overview.png?raw=true)



## Software Quick Start

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment:

```sh
conda create -y -n recOrder python=3.10
conda activate recOrder
```

Install `recOrder-napari`:

```sh
pip install recOrder-napari
```

Open `napari` with `recOrder-napari`:

```sh
napari -w recOrder-napari
```

For more help, see [`recOrder`'s documentation](https://github.com/mehta-lab/recOrder/tree/main/docs). To install `recOrder` 
on a microscope, see the [microscope installation guide](https://github.com/mehta-lab/recOrder/blob/main/docs/microscope-installation-guide.md).

## Dataset

[Slides](https://doi.org/10.5281/zenodo.5135889) and a [dataset](https://doi.org/10.5281/zenodo.5178487) shared during a workshop on QLIPP and recOrder can be found on Zenodo, and the napari plugin's sample contributions (`File > Open Sample > recOrder-napari` in napari).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)