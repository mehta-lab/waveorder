# waveorder

[![Python package index](https://img.shields.io/pypi/v/waveorder.svg)](https://pypi.org/project/waveorder)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/waveorder.svg)](https://pypistats.org/packages/waveorder)
[![Total downloads](https://pepy.tech/badge/waveorder)](https://pepy.tech/project/waveorder)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/mehta-lab/waveorder)](https://github.com/mehta-lab/waveorder/graphs/contributors)
![GitHub Repo stars](https://img.shields.io/github/stars/mehta-lab/waveorder)
![GitHub forks](https://img.shields.io/github/forks/mehta-lab/waveorder)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveorder)


This computational imaging library enables wave-optical simulation and reconstruction of optical properties that report microscopic architectural order.

## Computational label-agnostic imaging

https://github.com/user-attachments/assets/4f9969e5-94ce-4e08-9f30-68314a905db6

`waveorder` enables simulations and reconstructions of label-agnostic microscopy data as described in the following [preprint](https://arxiv.org/abs/2412.09775)
<details>	
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

Specifically, `waveorder` enables simulation and reconstruction of 2D or 3D:

1. __phase, projected retardance, and in-plane orientation__ from a polarization-diverse volumetric brightfield acquisition ([QLIPP](https://elifesciences.org/articles/55502)),

2. __phase__ from a volumetric brightfield acquisition ([2D phase](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-28-8566)/[3D phase](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-1-a205)),

3. __phase__ from an illumination-diverse volumetric acquisition ([2D](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-23-9-11394&id=315599)/[3D](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-10-3940&id=349951) differential phase contrast),

4. __fluorescence density__ from a widefield volumetric fluorescence acquisition (fluorescence deconvolution).  


## Permittivity tensor imaging 

Additionally, `waveorder` enabled the development of a new label-free imaging method, __permittivity tensor imaging (PTI)__, that measures density and  3D orientation of biomolecules with diffraction-limited resolution. These measurements are reconstructed from polarization-resolved images acquired with a sequence of oblique illuminations.

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

PTI provides volumetric reconstructions of mean permittivity ($\propto$ material density), differential permittivity ($\propto$ material anisotropy), 3D orientation, and optic sign. The following figure summarizes PTI acquisition and reconstruction with a small optical section of the mouse brain tissue:

![Data_flow](https://github.com/mehta-lab/waveorder/blob/main/readme.png?raw=true)

## Examples
The [examples](https://github.com/mehta-lab/waveorder/tree/main/examples) illustrate simulations and reconstruction for 2D QLIPP, 3D phase from brightfield, and 2D/3D PTI methods.

If you are interested in deploying QLIPP or phase from brightbrield, or fluorescence deconvolution for label-agnostic imaging at scale, checkout our [napari plugin](https://www.napari-hub.org/plugins/recOrder-napari),  [`recOrder-napari`](https://github.com/mehta-lab/recOrder).

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

Use `waveorder` in your scripts:

```sh
python
>>> import waveorder
```

(Optional) Install napari, clone the repository, and run an example script:
```sh
pip install napari[all]
git clone https://github.com/mehta-lab/waveorder.git
python waveorder/examples/models/phase_thick_3d.py
```

(Optional) Download the repository, install `jupyter`, and experiment with the example notebooks

```sh
git clone https://github.com/mehta-lab/waveorder.git
pip install jupyter
jupyter notebook ./waveorder/examples/
```

(M1 users) `pytorch` has [incomplete GPU support](https://github.com/pytorch/pytorch/issues/77764),
so please use `export PYTORCH_ENABLE_MPS_FALLBACK=1`
to allow some operators to fallback to CPU
if you plan to use GPU acceleration for polarization reconstruction. 