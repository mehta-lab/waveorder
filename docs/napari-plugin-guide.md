# Napari Plugin Guide

## Launch `recOrder`
Activate the `recOrder` environment
```
conda activate recOrder
```
   
Launch `napari` with `recOrder`
```
napari -w recOrder-napari
```

## Calibration tab
The first step in the QLIPP process is to calibrate the liquid crystals. This process involves generating the polarization states and acquiring a background **on an empty FOV**, so begin by placing your sample on the stage and focusing on the surface of the coverslip or well. 

The light path must also be in **Kohler Illumination** to ensure uniform illumination of the sample. [Please follow these steps to setup Kohler illumination.](https://www.microscopyu.com/tutorials/kohler)

After pressing “Connect to MM” and choosing a directory (where the calibration metadata / background images will be stored), the first step in calibration is to input the illumination wavelength and decide on the swing to use. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/connect_to_mm.png)

### Choose a swing value
The swing value is the deviation away from circular illumination states. The larger the swing, the more elliptical the polarized states becomes, until finally reaching a linear state at `swing = 0.25`. Picking a swing is dependent on the anisotropy of the sample. We recommend

* ​Tissue Imaging: `swing = 0.1 - 0.05`
* Live or fixed Cells: `swing = 0.05 – 0.03`

We recommend starting with a swing of **0.1** for tissue samples and **0.05** for cells then reducing the swing to measure smaller structures.

### Choose an illumination scheme
The illumination scheme decides which polarization states to calibrate and use. We recommend sticking with the *4-State (Ext, 0, 60, 120)* scheme as it requires one less illumination state than the *5-State* scheme.

### Run the calibration
Once the above parameters are set, the user is ready for "Run Calibration"

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/run_calib.png)

The progress bar will show the progress of calibration, and it should take less than 2 minutes on most systems.

The plot shows the intensities over time during the optimization. One way to diagnose if calibration is going smoothly is to look at the shape of this plot. An example of an ideal plot is below:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/ideal_plot.png)

Once finished, you will get a calibration assessment and an extinction value. The calibration assessment will tell you if anything is incorrect with the light path or if calibration went awry. Extinctions gives you a metric for calibration quality—the higher the extinction, the cleaner the light path and the greater the sensitivity of QLIPP.

* Extinction 0 – 50:  Very poor. The alignment of the universal compensator may be off or the sample chamber may be highly birefringent. 

* Extinction 50-100: Okay extinction, could be okay for tissue imaging and strong anisotropic structures. Most likely not suitable for cell imaging

* Extinction 100-200: Good Extinction. These are the typical values we get on our microscopes.

* Extinction 200+: Excellent. Indicates a very well-aligned and clean light path and high sensitivity of the system.

### Optional: Load Calibration
If a user wants to use a previous calibration to acquire new data, then the "Load Calibration" button can be pressed.  It will direct you to select a *calibration_metadata.txt* file and these settings will be automatically updated in MicroManager.  recOrder will also collect a few images to update the extinction ratio to reflect the current conditions.  Once this has finished, a user can now acquire data as they normally would.  

This is quite useful if micromanager and/or  recOrder crashes. If the sample and imaging setup haven't changed, it is safe to use a past calibration. Otherwise, if a new sample is used or some microscope components are changed, we recommend performing a new calibration.

### Optional: Calculate Extinction
This is a useful feature to see if the extinction level varies as you move around the sample.  Sometimes there can be local variations present in the sample which can cause slightly different perturbations to the polarization state.  If the extinction level varies dramatically across the sample, it is worthwhile to calibrate and acquire background images as close to the area in which you will be imaging as possible.

### Capture Background
The next important step in the calibration process. This will later serve in reconstruction to correct for any local and global background anisotropy. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/cap_bg.png)

Choose the name of the folder to which to save your background images (will be placed into the save directory chosen at the beginning). Choose the number of images to average, 20 and below is generally good.  The background image results will then be displayed in the napari window.  It is normal to see some level of background retardance and orientation bias-- this can be corrected with the background correction step in reconstruction.  Examples of this display are below.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/bg_example.png)

### Advanced Tab
The advanced tab gives the user a log output which can be useful for debugging purposes. There is a log level “debugging” which serves as a verbose output. Look here for any hints as to what may have gone wrong during calibration or acquisition.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/advanced.png)

### Example of Successful Calibration
Below is an example of a successful calibration with a reasonable extinction value:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/calib_finished.png)

## Acquisition / Reconstruction Tab
This acquisition module is designed to take single image volumes for both phase and birefringence measurements and allows the user to test the outcome of their calibration with small snapshots or volumes. We recommend this acquisition mode for quick testing and the Micromanager MDA acquisition for high-throughput data collection.

### Save Path
User specifies the directory in which the images will be saved.

### Acquisition Settings
*Z Start, Z End, Z Step* specify the relative z-parameters to use for acquiring an image volume. Values are in the default units of the stage, typically in microns. The center slice will be the current position of the z-stage

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/acquisition.png)

Ex. for a 20 um thick cell, the user would first focus in the middle of the cell and then set the following parameters: 

* `Z Start = -12`
* `Z End: 12`
* `Z Step: 0.25` 

For phase reconstruction, the stack should have sufficient defocus along the top and bottom of the stack. The reconstruction algorithm uses the defocus information to more accurately reconstruct phase.

User can then choose whether they want to acquire a 2D or 3D Birefringence/Phase stack. Note that even for a 2D phase stack, a full image volume is required for reconstruction purposes.

### Reconstruction Settings 
These settings are solely for reconstructing the acquired image / image volume. The *Phase Only* parameters are only needed for reconstructing phase. The user is also able to specify the use of a GPU for reconstruction (requires CuPy / CudaToolKit) if present, otherwise leave blank.

Explanation of background correction methods:
  
* None: No background correction is performed. 
* Measured: Corrects sample images with a background image acquired at an empty field of view, loaded from "Background Path". 
* Estimated: Estimate the sample background by fitting a 2D surface to the sample images. Works well when structures are spatially distributed across the field of view and a clear background is unavailable.
* Measured + Estimated: Applies "Measured" background correction and then "Estimated" background correction. Use to remove residual background after the sample retardance is corrected with measured background.

An explanation of phase reconstruction parameters:
 
* Wavelength (nm): Wavelength used for illumination
* Objective NA: Numerical Aperture of Objective, typically found next to magnification
* Condenser NA: Numerical Aperture of Condenser
* Magnification: Magnfication of the objective
* Camera Pixel Size: Pixel size of the camera in microns (ex. 6.5)
* RI of Obj Media: Refractive index of the objective media. Defaults to air (1.003). Typical values also include 1.512 (oil) or 1.473 (glycerol).
* Z Padding: The number of slices to pad on either end of the stack in order to correct for edge reflection artifacts. Necessary if the sample is not fully out of focus on either end of the stack.

The acquired data will then be displayed in the `napari` window. Note that phase reconstruction is more computationally expensive and may take several minutes depending on your system.

Examples of acquiring 2D birefringence data (kidney tissue) with this snap method are below:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/acq_finished.png)