# WAVEORDER
[![Python package index](https://img.shields.io/pypi/v/waveorder.svg)](https://pypi.org/project/waveorder)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/waveorder.svg)](https://pypistats.org/packages/waveorder)
[![Total downloads](https://pepy.tech/badge/waveorder)](https://pepy.tech/project/waveorder)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/mehta-lab/waveorder)](https://github.com/mehta-lab/waveorder/graphs/contributors)
![GitHub Repo stars](https://img.shields.io/github/stars/mehta-lab/waveorder)
![GitHub forks](https://img.shields.io/github/forks/mehta-lab/waveorder)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveorder)

Label-agnostic computational microscopy of architectural order.

# Overview

`waveorder` is a generalist framework for label-agnostic computational microscopy of architectural order, i.e., density, alignment, and orientation of biomolecules with a resolution down to the diffraction limit. The framework implements wave-optical simulations and corresponding reconstruction algorithms for diverse label-free and fluorescence computational imaging methods that enable large-scale image-based profiling of dynamic cell systems. Our goal is to provide unified, accurate, modular, and user-friendly implementations of computational microscopy methods for dynamic imaging across the scales of organelles, cells, and tissues.


The framework is described in the following [preprint](https://arxiv.org/abs/2412.09775).

https://github.com/user-attachments/assets/4f9969e5-94ce-4e08-9f30-68314a905db6

<details>
`waveorder` enables simulations and reconstructions of label-agnostic microscopy data as described in the following [preprint](https://arxiv.org/abs/2412.09775)
<summary> Chandler et al. 2024 </summary>
<pre><code>
@article{chandler_2024,
    author = {Chandler, Talon and Hirata-Miyasaki, Eduardo and Ivanov, Ivan E. and Liu, Ziwen and Sundarraman, Deepika and Ryan, Allyson Quinn and Jacobo, Adrian and Balla, Keir and Mehta, Shalin B.},
	title = {waveOrder: generalist framework for label-agnostic computational microscopy},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.09775},
	doi = {10.48550/arXiv.2412.09775}
}
</code></pre>
</details>

# Computational Microscopy Methods

The key feature of `waveorder` is that it provides a unified interface for simulating and reconstructing data for many one-photon (single-scattering based) computational microscopy methods, summarized below.

## Label-free microscopy

### Quantitative label-free imaging with phase and polarization (QLIPP)

Acquisition, calibration, background correction, reconstruction, and applications of QLIPP are described in the following [E-Life Paper](https://elifesciences.org/articles/55502):

[![Unveiling the invisible](https://github.com/mehta-lab/recOrder/blob/main/docs/images/comms_video_screenshot.png?raw=true)](https://www.youtube.com/watch?v=JEZAaPeZhck)

<details>
<summary> Guo et al. 2020 </summary>
<pre><code>
@article{guo_2020,
	author = {Guo, Syuan-Ming and Yeh, Li-Hao and Folkesson, Jenny and Ivanov, Ivan E. and Krishnan, Anitha P. and Keefe, Matthew G. and Hashemi, Ezzat and Shin, David and Chhun, Bryant B. and Cho, Nathan H. and Leonetti, Manuel D. and Han, May H. and Nowakowski, Tomasz J. and Mehta, Shalin B.},
	title = {Revealing architectural order with quantitative label-free imaging and deep learning},
	journal = {eLife},
	volume = {9},
	pages = {e55502},
	year = {2020},
	doi = {10.7554/eLife.55502}
}
</code></pre>
</details>

### Permittivity tensor imaging (PTI)

PTI provides volumetric reconstructions of mean permittivity ($\propto$ material density), differential permittivity ($\propto$ material anisotropy), 3D orientation, and optic sign. The following figure summarizes PTI acquisition and reconstruction with a small optical section of the mouse brain tissue:

![Data_flow](https://github.com/mehta-lab/waveorder/blob/main/readme.png?raw=true)

The acquisition, calibration, background correction, reconstruction, and applications of PTI are described in the following [paper](https://doi.org/10.1101/2020.12.15.422951) published in Nature Methods:

<details>
<summary> Yeh et al. 2024 </summary>
<pre><code>
@article{yeh_2024,
	author = {Yeh, Li-Hao and Ivanov, Ivan E. and Chandler, Talon and Byrum, Janie R. and Chhun, Bryant B. and Guo, Syuan-Ming and Foltz, Cameron and Hashemi, Ezzat and Perez-Bermejo, Juan A. and Wang, Huijun and Yu, Yanhao and Kazansky, Peter G. and Conklin, Bruce R. and Han, May H. and Mehta, Shalin B.},
	title = {Permittivity tensor imaging: modular label-free imaging of 3D dry mass and 3D orientation at high resolution},
	journal = {Nature Methods},
	volume = {21},
	number = {7},
	pages = {1257--1274},
	year = {2024},
	month = jul,
	issn = {1548-7105},
	publisher = {Nature Publishing Group},
	doi = {10.1038/s41592-024-02291-w}
}
</code></pre>
</details>


### Quantitative phase imaging (QPI) from defocus
__phase__ from a volumetric brightfield acquisition ([2D phase](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-28-8566)/[3D phase](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-1-a205))

### QPI with differential phase contrast
 __phase__ from an illumination-diverse volumetric acquisition ([2D](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-23-9-11394&id=315599)/[3D](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-10-3940&id=349951) differential phase contrast)

## Fluorescence microscopy

### Widefield and confocal deconvolution microscopy
__fluorescence density__ from a widefield  and confocalvolumetric fluorescence acquisition.

### Oblique plane light-sheet microscopy
__fluorescence density__ from oblique plane light-sheet microscopy.



## Citation

Please cite this repository, along with the relevant preprint or paper, if you use or adapt this code. The citation information can be found by clicking "Cite this repository" button in the About section in the right sidebar.

## Installation

Create a virtual environment:

```sh
conda create -y -n waveorder python=3.10
conda activate waveorder
```

Install `waveorder` from PyPI:

```sh
pip install waveorder
```

(Optional) Install all visualization dependencies (napari, jupyter), clone the repository, and run an example script:
```sh
pip install waveorder[all]
git clone https://github.com/mehta-lab/waveorder.git
python waveorder/examples/models/phase_thick_3d.py
```

(M1 users) `pytorch` has [incomplete GPU support](https://github.com/pytorch/pytorch/issues/77764),
so please use `export PYTORCH_ENABLE_MPS_FALLBACK=1`
to allow some operators to fallback to CPU if you plan to use GPU acceleration for polarization reconstruction.


## Examples
The [examples](https://github.com/mehta-lab/waveorder/tree/main/docs/examples) illustrate simulations and reconstruction for 2D QLIPP, 3D phase from brightfield, and 2D/3D PTI methods.
