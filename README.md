# waveorder
This package enables wave optical simulations and deconvolution of optical properties that report microscopic architectural order.

This vectorial wave simulator/reconstructor enabled the development of a new label-free imaging method, __permittivity tensor imaging (PTI)__, that measures 3D distribution of material density and material orientation ( 3D orientation of the symmetry axis, degree of anisotropy, and optic sign) of specimens from polarization-resolved images acquired with multiple oblique illuminations.

The acquisition, calibration, background correction, reconstruction, and applications of PTI are described in the following [preprint](https://doi.org/10.1101/2020.12.15.422951):

``` L.-H. Yeh, I. E. Ivanov, B. B. Chhun, S.-M. Guo, E. Hashemi, J. R. Byrum, J. A. Pérez-Bermejo, H. Wang, Y. Yu, P. G. Kazansky, B. R. Conklin, M. H. Han, and S. B. Mehta, "uPTI: uniaxial permittivity tensor imaging of intrinsic density and anisotropy," bioRxiv 2020.12.15.422951 (2020).```

Please cite this repository, along with the relevant preprint or paper, if you use or adapt this code.

PTI enables reconstruction of the material's permittivity, 3D orientation, and optic sign from a polarization-diverse and illumination-diverse acquisition. Following figure summarizes how the acquisition and reconstructions with `waveorder` work using a small field of view in a mouse brain tissue section: 

![Data_flow](Fig_Readme.png)

When the acquisition is polarization-diverse, illumination-diverse, and depth-diverse, `waveorder` can reconstruct the above measurements across volume. 

In addition to PTI, `waveorder` also enables simulations and reconstructions of subsets of label-free measurements with subsets of acquired dimensions.

1. Reconstruction of 2D/3D phase, projected retardance, and in-plane orientation from a brightfield, polarization-diverse, and depth-diverse acquisition ([QLIPP](https://elifesciences.org/articles/55502))

2. Reconstruction of 2D/3D phase from a brightfield, depth-diverse acquisition ([2D](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-28-8566)/[3D (PODT)](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-1-a205) phase)

3. Reconstruction 2D/3D phase from an illumination-diverse and depth-diverse acquisition ([2D](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-23-9-11394&id=315599)/[3D](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-10-3940&id=349951) differential phase contrast)

Currently, the [example](examples/) notebooks contain simulations for 2D QLIPP, 3D PODT, and 2D/3D PTI. There is also a notebook demonstrating the reconstruction of the experimental data of 3D PTI (data will be uploaded upon acceptance of the manuscript).

If you are interested in using QLIPP or PODT for label-free imaging at scale, checkout our [napari plugin](https://www.napari-hub.org/plugins/recOrder-napari),  [`recOrder-napari`](https://github.com/mehta-lab/recOrder).
## Installation

### Create a new conda environment
Install conda package management system by installing anaconda or miniconda ([link](https://conda.io/)). 

1) Creating a conda environment dedicated to `waveorder` will avoid version conflicts among packages required by `waveorder` and packages required by other python software.
>```buildoutcfg
>conda create -n <your-environment-name> python=3.7
>conda activate <your-environment-name> (or source activate <your-environment-name>)
>```

2) Then, install jupyter notebook with
>```buildoutcfg
>conda install jupyter
>```
    
### Install `waveorder` and required packages
Install the git version control system git : [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

1) Use git to clone this repository to your current directory:
>```buildoutcfg
>git clone https://github.com/mehta-lab/waveorder.git
>```

2) Then, you can install dependencies via pip (python index package) <br>
  
>    If you are running waveorder on your own machine, <br>
>
>    a) navigate to the cloned repository:
>
>    ```buildoutcfg
>    cd waveorder
>    ```
>    <br>
>    b) install python library dependencies:
>
>    ```buildoutcfg
>    pip install -r requirements.txt
>    ```
>    <br>

3) Create a symbolic library link with setup.py:
>
>```buildoutcfg
>python setup.py develop
>```

The installation time is less than 10 minutes on a normal desktop computer.

*`waveorder` supports NVIDIA GPU computation through cupy package, please follow [here](https://github.com/cupy/cupy) for installation (check cupy is properly installed by ```import cupy```). To enable gpu processing, set ```use_gpu=True``` when initializing the simulator/reconstructor class.*


## Usage and example

In the following, we demonstrate how to run `waveorder` for simulation and reconstruction. <br>

1) In the terminal, switch to the environment with waveorder installed 
>  ```buildoutcfg
>  conda activate <your-environment-name>
>  ```

2) Navigate to the repository folder:
>  ```buildoutcfg
>  cd waveorder/example
>  ```

3) Open jupyter notebook or lab to run the simulation/reconstruction notebook in the folder:
>  ```buildoutcfg
>  jupyter notebook
>  ```
We recommend installing `cupy` before running PTI simulation because PTI computation takes up more resources. 3D PTI simulation with array size of (200, 200, 100) takes 20 minutes and the reconstruction of the same-size array takes 10 minutes on a NVIDIA Titan Xp GPU.
    
