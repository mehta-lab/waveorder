import time
import logging
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

# =========== Methods to set/get LC properties ===============

def set_lc_waves(mmc, waves: float, device_property: str):
    """
    Puts a retardance value for either LCA or LCB

    Parameters
    ----------
    mmc:                (object) MM Core object
    waves:              (float) Retardance to set in fraction of wavelength
    device_property:    (str) 'Retardance LC-A [in waves]' or 'Retardance LC-B [in waves]'

    Returns
    -------

    """
    if waves > 1.6 or waves < 0.001:
        raise ValueError("waves must be float, greater than 0.001 and less than 1.6")
    mmc.setProperty('MeadowlarkLcOpenSource', device_property, str(waves))
    time.sleep(20/1000)

def set_lc_volts(mmc, volts: float, device_property: str):
    """
    Puts a retardance value for either LCA or LCB

    Parameters
    ----------
    mmc:                (object) MM Core object
    waves:              (float) Voltage to set on triggerscope DAC, 0-5 V
    device_property:    (str) 'LCA' or 'LCB'

    Returns
    -------

    """
    if volts > 5.0 or volts < 0.0:
        raise ValueError("Voltages must be float, greater than 0 and less than 5")

    mmc.setProperty(device_property, 'Volts', str(volts))
    time.sleep(0.4) # 10 ms?

def get_lc_waves(mmc, device_property: str) -> float:
    """
    Get LC Retardance Value

    Parameters
    ----------
    mmc:                (object) MM Core object
    device_property:    (str) 'LCA' or 'LCB'

    Returns
    -------
    retardance:         (float) Retardance of desires LC in fraction of wavelength

    """
    return float(mmc.getProperty('MeadowlarkLcOpenSource', device_property))

def get_lc_volts(mmc, device_property):
    return float(mmc.getProperty(device_property, 'Volts'))

def define_lc_state_volts(mmc, group, state, lca, lcb, lca_dac, lcb_dac):
    """
    Write specific LC state to the register, corresponds to 'State{i}' in MM config

    Parameters
    ----------
    mmc:                (object) MM Core object
    state:              (string) State upon which LC values will be saved to.  'State{i}'
    lca:                (float) Retardance of desires LC in fraction of wavelength
    lcb:                (float) Retardance of desires LC in fraction of wavelength
    PROPERTIES:         (dict) Properties dictionary which shortcuts MM device property names

    Returns
    -------

    """
    set_lc_volts(mmc, lca, lca_dac)
    set_lc_volts(mmc, lcb, lcb_dac)

    logging.debug("setting LCA = "+str(lca))
    logging.debug("setting LCB = "+str(lcb))

    mmc.defineConfig(group, state, lca_dac, 'Volts', str(lca))
    mmc.defineConfig(group, state, lcb_dac, 'Volts', str(lcb))
    mmc.waitForConfig(group, state)

def define_lc_state(mmc, state, lca, lcb, PROPERTIES: dict):
    """
    Write specific LC state to the register, corresponds to 'State{i}' in MM config

    Parameters
    ----------
    mmc:                (object) MM Core object
    state:              (string) State upon which LC values will be saved to.  'State{i}'
    lca:                (float) Retardance of desires LC in fraction of wavelength
    lcb:                (float) Retardance of desires LC in fraction of wavelength
    PROPERTIES:         (dict) Properties dictionary which shortcuts MM device property names

    Returns
    -------

    """
    set_lc_waves(mmc, lca, PROPERTIES['LCA'])
    set_lc_waves(mmc, lcb, PROPERTIES['LCB'])

    logging.debug("setting LCA = "+str(lca))
    logging.debug("setting LCB = "+str(lcb))

    mmc.setProperty('MeadowlarkLcOpenSource', PROPERTIES[state], 0)
    mmc.waitForDevice('MeadowlarkLcOpenSource')


def set_lc_state(mmc, group: str, state: str):
    """
    Change to the specific LC State

    Parameters
    ----------
    mmc:                (object) MM Core object
    state:              (string) State to switch to.  'State{i}'

    """
    try:
        mmc.setConfig(group, state)
    except ValueError:
        print('ERROR: No MicroManager Config Group/Preset named Channel')
    time.sleep(20/1000) # delay for LC settle time