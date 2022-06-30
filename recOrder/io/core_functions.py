import time
import numpy as np


def snap_image(mmc):
    """
    Snap and return an image through pycromanager ZMQ

    Parameters
    ----------
    mmc:        (object) MM core object

    Returns
    -------
    image:      (array) 1D-Array of length Y * X

    """

    mmc.snapImage()
    time.sleep(0.3) # sleep after snap to make sure the image we grab is the correct one

    return mmc.getImage()


def snap_and_get_image(snap_manager):
    """
    Snap and get image using Snap Live Window Manager + transfer of ZMQ

    Parameters
    ----------
    snap_manager:   (object) MM Snap Live Window object

    Returns
    -------
    image:          (array) 2D array of size (Y, X)

    """
    snap_manager.snap(True)
    time.sleep(0.3) # sleep after snap to make sure the image we grab is the correct one

    # get pixels + dimensions
    height = snap_manager.getDisplay().getDisplayedImages().get(0).getHeight()
    width = snap_manager.getDisplay().getDisplayedImages().get(0).getWidth()
    array = snap_manager.getDisplay().getDisplayedImages().get(0).getRawPixels()

    return np.reshape(array, (height, width))


def snap_and_average(snap_manager, display=True):
    """
    Snap an image with Snap Live manager + grab only the mean (computed in java)

    Parameters
    ----------
    snap_manager:   (object) MM Snap Live Window object
    display:        (bool) Whether to show the snap on the Snap Live Window in MM

    Returns
    -------
    mean:           (float) mean of snapped image

    """

    snap_manager.snap(display)
    time.sleep(0.3)  # sleep after snap to make sure the image we grab is the correct one

    return snap_manager.getDisplay().getImagePlus().getStatistics().umean


def set_lc_waves(mmc, device_property: tuple, value: float):
    """
    Set retardance in waves for LC in device_property

    Parameters
    ----------
    mmc : object
        MM Core object
    device_property : tuple
        (device_name, property_name) set
    value : float
        Retardance to set as fraction of a wavelength

    Returns
    -------

    """
    device_name = device_property[0]
    prop_name = device_property[1]

    if value > 1.6 or value < 0.001:
        raise ValueError(f"Requested retardance value is {value} waves. "
                         f"Retardance must be greater than 0.001 and less than 1.6 waves.")

    mmc.setProperty(device_name, prop_name, str(value))
    time.sleep(20/1000)


def set_lc_voltage(mmc, device_property: tuple, value: float):
    """
    Set LC retardance by specifying LC voltage

    Parameters
    ----------
    mmc : object
        MM Core object
    device_property : tuple
        (device_name, property_name) set
    value : float
        LC voltage in volts. Applied voltage is limited to 20V

    Returns
    -------

    """
    device_name = device_property[0]
    prop_name = device_property[1]

    if value > 20.0 or value < 0.0:
        raise ValueError(f"Requested LC voltage is {value} V. "
                         f"LC voltage must be greater than 0.0 and less than 20.0 V.")

    mmc.setProperty(device_name, prop_name, str(value))
    time.sleep(20 / 1000)


def set_lc_daq(mmc, device_property: tuple, value: float):
    """
    Set LC retardance based on DAQ output

    Parameters
    ----------
    mmc : object
        MM Core object
    device_property : tuple
        (device_name, property_name) set
    value : float
        DAQ output voltage in volts. DAQ output must be in 0-5V range

    Returns
    -------

    """
    device_name = device_property[0]
    prop_name = device_property[1]

    if value > 5.0 or value < 0.0:
        raise ValueError("DAC voltage must be greater than 0.0 and less than 5.0")

    mmc.setProperty(device_name, prop_name, str(value))
    time.sleep(20 / 1000)


def get_lc(mmc, device_property: tuple):
    """
    Get LC state in the native units of the device property

    Parameters
    ----------
    mmc : object
        MM Core object
    device_property : tuple
        (device_name, property_name) set

    Returns
    -------

    """

    device_name = device_property[0]
    prop_name = device_property[1]

    val = float(mmc.getProperty(device_name, prop_name))
    return val


def define_meadowlark_state(mmc, device_property: tuple):
    """
    Defines pallet element in the Meadowlark device adapter for the given state.
    Make sure LC values for this state are set before calling this function

    Parameters
    ----------
    mmc : object
        MM Core object
    device_property : tuple
        (device_name, property_name) set, e.g.
        ('MeadowlarkLC', 'Pal. elem. 00; enter 0 to define; 1 to activate')

    Returns
    -------

    """

    device_name = device_property[0]
    prop_name = device_property[1]

    # define LC state
    # setting pallet elements to 0 defines LC state
    mmc.setProperty(device_name, prop_name, 0)
    mmc.waitForDevice(device_name)


def define_config_state(mmc, group: str, config: str, device_properties: list, values: list):
    """
    Define config state by specifying the values for all device properties in this config

    Parameters
    ----------
    mmc : object
        MM Core object
    group : str
        Name of config group
    config : str
        Name of config, e.g. State0
    device_properties: list
        List of (device_name, property_name) tuples in config
    values: list
        List of matching device property values

    Returns
    -------

    """

    for device_property, value in zip(device_properties, values):
        device_name = device_property[0]
        prop_name = device_property[1]
        mmc.defineConfig(group, config, device_name, prop_name, str(value))
    mmc.waitForConfig(group, config)


def set_lc_state(mmc, group: str, config: str):
    """
    Change to the specific LC State

    Parameters
    ----------
    mmc : object
        MM Core object
    group : str
        Name of config group
    config : str
        Name of config, e.g. State0

    """

    mmc.setConfig(group, config)
    time.sleep(20/1000)  # delay for LC settle time