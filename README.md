# recOrder
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/recOrder-napari)
[![Downloads](https://pepy.tech/badge/recOrder-napari)](https://pepy.tech/project/recOrder-napari)
[![Python package index](https://img.shields.io/pypi/v/recOrder-napari.svg)](https://pypi.org/project/recOrder-napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This package provides a napari plugin for quantitative label-free microscopy.

In this repository you will find python tools and a napari plugin that allow the user to calibrate microscope hardware, acquire multi-modal data, reconstruct density and anisotropy, and visualize the results.

The acquisition, calibration, background correction, reconstruction, and applications of QLIPP (quantitative label-free imaging with phase and polarization)  are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

```bibtex
Syuan-Ming Guo, Li-Hao Yeh, Jenny Folkesson, Ivan E Ivanov, Anitha P Krishnan, Matthew G Keefe, Ezzat Hashemi, David Shin, Bryant B Chhun, Nathan H Cho, Manuel D Leonetti, May H Han, Tomasz J Nowakowski, Shalin B Mehta, "Revealing architectural order with quantitative label-free imaging and deep learning," eLife 2020;9:e55502 DOI: 10.7554/eLife.55502 (2020).
```

`recOrder` is to be used alongside a conventional widefield microscope fitted with a universal polarizer (Panel A below).  The universal polarizer allows for the collection of label-free information including the intrinsic anisotropy of the sample and its relative phase (density). These measurements are collected by acquiring data under calibrated, polarization-diverse illumination followed by a computational reconstruction.  The overall structure of `recOrder` is shown in Panel B, highlighting the two different usage modes and their features: graphical user interface (GUI) through a napari plugin, and a command line interface (CLI).

![Flow Chart](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_Fig1_Overview.png?raw=true)

## Dataset

[Slides](https://doi.org/10.5281/zenodo.5135889) and a [dataset](https://doi.org/10.5281/zenodo.5178487) shared during a workshop on QLIPP and recOrder can be found on Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)

## Quick Start

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment:

```sh
conda create -y -n recOrder python=3.9
conda activate recOrder
```

> *Apple Silicon users please use*:
>
> ```sh
> CONDA_SUBDIR=osx-64 conda create -y -n recOrder python=3.9
> conda activate recOrder
> ```

Install `recOrder-napari`:

```sh
pip install recOrder-napari
```

Open `napari` with `recOrder-napari`:

```sh
napari -w recOrder-napari
```

View command-line help by running

```sh
recOrder.help
```

For more help, see [`recOrder`s documentation](./docs).
