# Napari Plugin Guide

## Launching `recOrder`

Activate the `recOrder` environment
```
conda activate recOrder
```
   
Launch `napari` with `recOrder`
```
napari -w recOrder-napari
```

## `recOrder` Calibration

The first step in the QLIPP process is to calibrate the universal polarizer. This process involves generating the polarization states and acquiring a background **on an empty FOV**.  The light path must also be in **Kohler Illumination** in order to ensure uniform illumination of the sample. Steps on Kohler illumination can be found here https://www.microscopyu.com/tutorials/kohler.

More info on the details of the calibration process and use of the plugin can be found on Zenodo which were taken from a workshop at the Chan-Zuckerberg Biohub in June 2021.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5178487.svg)](https://doi.org/10.5281/zenodo.5178487)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5135889.svg)](https://doi.org/10.5281/zenodo.5135889)

Turn on online mode by clicking "Switch To Online" button.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/switch_online.png)

After pressing “Connect to MM” and choosing a directory (where the calibration metadata / background images will be stored), the first step in calibration is to input the illumination wavelength and decide on the swing to use. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/connect_to_mm.png)

### Deciding Swing

Swing can be thought of as the deviation away from a perfectly circular illumination state. The greater the swing, the more elliptical the state becomes, until finally reaching a linear state at *swing = 0.25*. Picking a swing is dependent on the anisotropy of the sample. Typical swing values range from *0.1 to 0.03.* Follow the guidelines below for an ideal swing setting.

 

​				<u>Live or fixed Cells:</u>     swing = 0.05 – 0.03

​				 <u>Tissue Imaging</u>:        swing = 0.1 - 0.05

 

We recommend using a swing of **0.1**  for most tissue samples and **0.05** for cells.

 

### Illumination Scheme

The illumination scheme decides which polarization states to calibrate and use. We recommend sticking with the *4-State (Ext, 0, 60, 120)* scheme as it requires one less illumination state than the *5-State* scheme.

 

### Use Cropped ROI

Calibration should be run on an empty field-of-view or “background” FOV in order to ensure that we are only compensating for the optical effects of the microscope and sample chamber. If you cannot find a fully empty FOV, you can draw a bounding box, or ROI, on the “Live Window” in MicroManager and check the *Use Cropped ROI* box



### Running Calibration

Once the above parameters are set, the user is ready to run the calibration and the button can be pressed.



![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/run_calib.png)



The progress bar will show the progress of calibration, and it should take less than 2 minutes on most systems.



The plot shows the Intensities over time during the optimization. One way to diagnose if calibration is going smoothly is to look at the shape of this plot. An example of an ideal plot is below:



![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/ideal_plot.png)

 

Once finished, you will get a calibration assessment and an extinction value. The calibration assessment will tell you if anything is incorrect with the light path or if calibration went awry. Extinctions gives you a metric for calibration quality—the higher the extinction, the cleaner the light path and the greater the sensitivity of QLIPP.

 

> *<u>Extinction 0 – 50:</u>*  Very poor. The alignment of the universal compensator may be off or the sample chamber may be highly birefringent.

 

> <u>*Extinction 50-100:*</u> Okay extinction, could be okay for tissue imaging and strong anisotropic structures. Most likely not suitable for cell imaging

 

> *<u>Extinction 100-200:</u>* Good Extinction. These are the typical values we get on our microscopes.

 

> <u>*Extinction 200+:*</u> Phenomenal. Indicates a very well-aligned and clean light path and high sensitivity of the system.



### Load Calibration*

If a user wants to use a previous calibration to acquire new data, then the "Load Calibration" button can be pressed.  It will direct you to select a *calibration_metadata.txt* file and these settings will be automatically updated in MicroManager.  recOrder will also collect a few images to update the extinction ratio to reflect the current conditions.  Once this has finished, a user can now acquire data as they normally would.  



*This is quite useful for micromanager crashes and potential recOrder crashes.  If nothing about the sample / imaging setup has changed, it is safe to use a past calibration.  Otherwise, if a new sample is used or some microscope components are changed, it is recommended to perform a new calibration.



### Calculate Extinction

This is a useful feature to see if the extinction level varies as you move around the sample.  Sometimes there can be local variations present in the sample which can cause slightly different perturbations to the polarization state.  If the extinction level varies dramatically across the sample, it is worthwhile to calibrate and acquire background images as close to the area in which you will be imaging as possible.



### Capturing Background

The next important step in the calibration process. This will later serve in reconstruction to correct for any local and global background anisotropy. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/cap_bg.png)

Choose the name of the folder to which to save your background images (will be placed into the save directory chosen at the beginning). Choose the number of images to average, 20 and below is generally good.  The background image results will then be displayed in the napari window.  It is normal to see some level of background retardance and orientation bias-- this can be corrected with the background correction step in reconstruction.  Examples of this display are below.

*NOTE: If you wish the capture multiple background sets, please change the folder name in between captures as specifying the same name will overwrite previous data.*

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/bg_example.png)

### Advanced Tab

The advanced tab gives the user a log output which can be useful for debugging purposes. There is a log level “debugging” which serves as a verbose output. Look here for any hints as to what may have gone wrong during calibration or acquisition.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/advanced.png)


### Example of Successful Calibration

Below is an example of a successful calibration with a reasonable extinction value:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/calib_finished.png)

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## recOrder Acquisition

This acquisition module is designed to take single image volumes for both phase and birefringence measurements and allows the user to test the outcome of their calibration with small snapshots or volumes. *It should not be used as the main acquisition method for an experiment, please use the Micromanager MDA Acquisition instead.* 

### Save Path

User specifies the directory in which the images will be saved. Needed if the *Save Images* box is checked.



### Acquisition Settings

*Z Start, Z End, Z Step* specify the relative z-parameters to use for acquiring an image volume. Values are in the default units of the stage, typically in microns. The center slice will be the current position of the z-stage

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/acquisition.png)

Ex. for a 20 um thick cell, the user would first focus in the middle of the cell and then set the following parameters: 

​									`Z Start: 	-12` 

​									 `Z End:	 12`

​									`Z Step:	 0.25` 

For phase reconstruction, the stack should have sufficient defocus along the top and bottom of the stack. The reconstruction algorithm uses the defocus information to more accurately reconstruct phase.

User can then choose whether they want to acquire a 2D or 3D Birefringence/Phase stack. Note that even for a 2D phase stack, a full image volume is required for reconstruction purposes.

### Reconstruction Settings 

These settings are solely for reconstructing the acquired image / image volume. The *Phase Only* parameters are only needed for reconstructing phase. The user is also able to specify the use of a GPU for reconstruction (requires CuPy / CudaToolKit) if present, otherwise leave blank.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/reconstruct_acq2.png)

Explanation of background correction methods:

> <u>*None:*</u> No Background correction is performed. Not necessary to specify a background folder.

> <u>*Global:*</u> A Global background correction is performed on every image by subtracting the background images from the sample in the stokes space.

> <u>*Local Fit**:*</u> A global background correction is formed and an additional estimation of local background is computed with a polynomial surface fit.  This is the preferred background correction method.

An explanation of phase reconstruction parameters:
 
> <u>*Wavelength (nm):*</u> Wavelength used for illumination [list or str]
> <u>*Objective NA:*</u> Numerical Aperture of Objective, typically found next to magnification
> <u>*Condenser NA:*</u> Numerical Aperture of Condenser
> <u>*Magnification:*</u> Magnfication of the objective
>  <u>*Camera Pixel Size:*</u> Pixel size of the camera in microns (ex. 6.5)
>  <u>*RI of Obj, Media:*</u> Refractive Index of the objective media. Defaults to air (1.003). Typical values also include 1.512 (oil) or 1.473 (glycerol).
>  <u>*Z Padding:*</u> The number of slices to pad on either end of the stack in order to correct for edge reflection artifacts. Necessary if the sample is not fully out of focus on either end of the stack.

The acquired data will then be displayed in the Napari window. Note that phase reconstruction is rather compute heavy and may take several minutes depending on your system.

Examples of Acquiring 2D birefringence data (Kidney Tissue) with this snap method are below:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/acq_finished.png)

________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


# recOrder Offline Reconstruction.

To use recOrder's offline reconstruction module you will need to switch to the offline mode by clicking the "Switch to Offline" button at the very top.  If you have just opened the plugin then it will start out in the offline mode.  The offline mode is pictured below:

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/recOrder_offline.png)

This plugin mimics the command line interface reconstruct functionality, but allows the user to enter in the configuration parameters in the GUI itself.  The user is also able to save these parameters to a new config file or load parameters into the app from an existing config file.  

As the data is being reconstructed, it will output the positions as different layers into napari.  This allows the user to see that data as it is being reconstructed and allows for quick interpretation of results.  Please see the workflow tutorial video in the tutorials section

