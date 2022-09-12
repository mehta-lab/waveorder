# Software Installation Guide

## User installation

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment  
```
conda create -y -n recOrder python=3.9
conda activate recOrder
```
Install `recOrder-napari`:
```
pip install recOrder-napari
```
Open `napari` with `recOrder-napari`:
```
napari -w recOrder-napari
```
View command-line help by running
```
recOrder.help
```

## Developer installation:

Install [`git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and `conda` (either [anaconda](https://www.anaconda.com/products/distribution) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Create a conda environment dedicated to `recOrder`:
```
conda create -y -n recOrder python=3.9
conda activate recOrder
```

Clone this repository:
```
git clone https://github.com/mehta-lab/recOrder.git
```

Install `recOrder` and its developer dependencies:
```
cd recOrder
pip install -e ".[dev]"
```

To acquire data via `Micromanager`, follow  [microscope installation guide](./microscope-installation-guide.md).

**Optional GPU**: `recOrder` supports NVIDIA GPU computation with the `cupy` package. Follow [these instructions](https://github.com/cupy/cupy) to install `cupy` and check its installation with ```import cupy```. To enable gpu processing, set ```use_gpu: True``` in the config files.

