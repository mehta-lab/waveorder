# Microscope Installation Guide

This guide will walk through a complete recOrder installation consisting of:
1. Installing and launching the latest stable version of `recOrder` via `pip`. 
2. Installing a compatible version of `MicroManager` and LC device drivers.
3. Connecting `recOrder` to `MicroManager` via a `pycromanager` connection.

Before you start you will need a machine with Windows 10, a Meadowlark DS5020 connected to a liquid crystal device, and a microscope system compatible with `Micromanager`. 

## Install recOrder software

(Optional but recommended) install [anaconda](https://www.anaconda.com/products/distribution) and create a virtual environment  
```
conda create -y -n recOrder python=3.9
conda activate recOrder
```

Install `recOrder`:
```
pip install recOrder-napari
```
Check your installation:
```
napari -w recOrder-napari
```
should launch napari (may take 15 seconds on a fresh installation) with the recOrder plugin in "Offline" mode. 
 
## Install and configure `Micromanager`

Install `Micromanager 2.0` nightly build `20220920` (https://micro-manager.org/Micro-Manager_Nightly_Builds). 

**Note:** We have tested recOrder with `20220920`, but most features will work with newer builds. We recommend testing a minimal installation with `20220920` before testing with a different nightly build or additional device drivers. 

Before launching `Micromanager`, download the Meadowlark device adapters and calibration files from the [release page](https://github.com/mehta-lab/recOrder/releases/) and place these three unzipped files into your `Micromanager` folder (likely `C:\Program Files\Micro-Manager` or similar). 

Launch `Micromanager`, open `Devices > Hardware Configuration Wizard...`, and add the `MeadowlarkLcOpenSource` device to your configuration. Confirm your installation by opening `Devices > Device Property Browser...` and confirming that `MeadowlarkLCOpenSource` properties appear. 

### Option 1 (recommended): Voltage-mode calibration installation
 Create a new channel group and add the `MeadowlarkLcOpenSource-Voltage (V) LC-A` and `MeadowlarkLcOpenSource-Voltage (V) LC-B` properties. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/create_group_voltage.png)

Add 5 presets to this group named `State0`, `State1`, `State2`, `State3`, and `State4`. You can set random voltages to add these presets, and `recOrder` will calibrate and set these voltages later.

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/create_preset_voltage.png)

### Option 2 (soon deprecated): retardance mode calibration installation

Create a new channel group and add the property `MeadowlarkLcOpenSource-String send to -`. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/create_group.png)

Add 5 presets to this group named `State0`, `State1`, `State2`, `State3`, and `State4` and set the corresponding preset values to `state0`, `state1`, `state2`, `state3`, `state4` in the `MeadowlarkLcOpenSource-String send to –`* property. 

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/create_preset.png)

### Enable port access

Finally, enable port access so that micromanager can communicate with recOrder through the `pycromanager` bridge. To do so open MM and navigate to `Tools > Options` and check the box that says `Run server on port 4827`

![](https://github.com/mehta-lab/recOrder/blob/main/docs/images/run_port.png)

## Connect `recOrder` to `Micromanager`

From the `recOrder` window, click `Switch to Online`. If you see `Success`, your installation is complete and you can [proceed to the napari plugin guide](./napari-plugin-guide.md). 

If you you see `Failed`, check that `Micromanager` is open, check that you've enabled `Run server on port 4827`. If the connection continues to fail, report an issue with you stack trace for support. 
