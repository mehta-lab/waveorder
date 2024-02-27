# waveorder

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveorder)
[![Downloads](https://pepy.tech/badge/waveorder)](https://pepy.tech/project/waveorder)
[![Python package index](https://img.shields.io/pypi/v/waveorder.svg)](https://pypi.org/project/waveorder)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This computational imaging library enables wave-optical simulation and reconstruction of optical properties that report microscopic architectural order.

## Computational label-free imaging

This vectorial wave simulator and reconstructor enabled the development of a new label-free imaging method, __permittivity tensor imaging (PTI)__, that measures density and  3D orientation of biomolecules with diffraction-limited resolution. These measurements are reconstructed from polarization-resolved images acquired with a sequence of oblique illuminations.

The acquisition, calibration, background correction, reconstruction, and applications of PTI are described in the following [preprint](https://doi.org/10.1101/2020.12.15.422951):

```bibtex
 L.-H. Yeh, I. E. Ivanov, B. B. Chhun, S.-M. Guo, E. Hashemi, J. R. Byrum, J. A. Pérez-Bermejo, H. Wang, Y. Yu, P. G. Kazansky, B. R. Conklin, M. H. Han, and S. B. Mehta, "uPTI: uniaxial permittivity tensor imaging of intrinsic density and anisotropy," bioRxiv 2020.12.15.422951 (2020).
 ```

In addition to PTI, `waveorder` enables simulations and reconstructions of subsets of label-free measurements with subsets of the acquired data:

1. Reconstruction of 2D or 3D phase, projected retardance, and in-plane orientation from a polarization-diverse volumetric brightfield acquisition ([QLIPP](https://elifesciences.org/articles/55502))

2. Reconstruction of 2D or 3D phase from a volumetric brightfield acquisition ([2D](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-28-8566)/[3D (PODT)](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-1-a205) phase)

3. Reconstruction of 2D or 3D phase from an illumination-diverse volumetric acquisition ([2D](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-23-9-11394&id=315599)/[3D](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-10-3940&id=349951) differential phase contrast)

PTI provides volumetric reconstructions of mean permittivity ($\propto$ material density), differential permittivity ($\propto$ material anisotropy), 3D orientation, and optic sign. The following figure summarizes PTI acquisition and reconstruction with a small optical section of the mouse brain tissue:

![Data_flow](https://github.com/mehta-lab/waveorder/blob/main/readme.png?raw=true)


The [examples](https://github.com/mehta-lab/waveorder/tree/main/examples) illustrate simulations and reconstruction for 2D QLIPP, 3D PODT, and 2D/3D PTI methods.

If you are interested in deploying QLIPP or PODT for label-free imaging at scale, checkout our [napari plugin](https://www.napari-hub.org/plugins/recOrder-napari),  [`recOrder-napari`](https://github.com/mehta-lab/recOrder).

## Correlative imaging

In addition to label-free reconstruction algorithms, `waveorder` also implements widefield fluorescence and fluorescence polarization reconstruction algorithms for correlative label-free and fluorescence imaging.

1. Correlative measurements of biomolecular density and orientation from polarization-diverse widefield imaging ([multimodal Instant PolScope](https://opg.optica.org/boe/fulltext.cfm?uri=boe-13-5-3102&id=472350))

We provide an [example notebook](https://github.com/mehta-lab/waveorder/blob/main/examples/documentation/fluorescence_deconvolution/fluorescence_deconv.ipynb) for widefield fluorescence deconvolution.

## Citation

Please cite this repository, along with the relevant preprint or paper, if you use or adapt this code. The citation information can be found by clicking "Cite this repository" button in the About section in the right sidebar.

## Installation

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment:

```sh
conda create -y -n waveorder python=3.9
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