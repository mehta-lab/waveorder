import time
import logging
import numpy as np

def snap_image(mmc):

    mmc.snapImage()

    return mmc.getImage()

def snap_and_get_image(snap_manager):
    snap_manager.snap(True)
    time.sleep(0.1)
    height = snap_manager.getDisplay().getDisplayedImages().get(0).getHeight()
    width = snap_manager.getDisplay().getDisplayedImages().get(0).getWidth()
    array = snap_manager.getDisplay().getDisplayedImages().get(0).getRawPixels()

    return np.reshape(array, (height,width))

def snap_and_average(snap_manager, display=True):

    snap_manager.snap(display)
    time.sleep(0.1)

    return snap_manager.getDisplay().getImagePlus().getStatistics().umean

# =========== Methods to set/get LC properties ===============

def set_lc(mmc, waves: float, device_property: str):
    """
    puts a value on LCA or LCB
    :param waves: float
        value in radians [0.001, 1.6]
    :param device_property: str
        'LCA' or 'LCB'
    :return: None
    """
    if waves > 1.6 or waves < 0.001:
        raise ValueError("waves must be float, greater than 0.001 and less than 1.6")
    mmc.setProperty('MeadowlarkLcOpenSource', device_property, str(waves))
#     mmc.waitForDevice('MeadowlarkLcOpenSource')
    time.sleep(20/1000)


def get_lc(mmc, device_property: str) -> float:
    """
    getter for LC value
    :param device_property: str
        'LCA' or 'LCB'
    :return: float
    """
    return float(mmc.getProperty('MeadowlarkLcOpenSource', device_property))


def define_lc_state(mmc, state, lca, lcb, PROPERTIES: dict):
    """
    defines the state based on current LCA - LCB settings
    :param device_property: str
        'State0', 'State1', 'State2' ....
    :return: None
    """
    set_lc(mmc, lca, PROPERTIES['LCA'])
    set_lc(mmc, lcb, PROPERTIES['LCB'])

    logging.debug("setting LCA = "+str(lca))
    logging.debug("setting LCB = "+str(lcb))

    mmc.setProperty('MeadowlarkLcOpenSource', PROPERTIES[state], 0)
    mmc.waitForDevice('MeadowlarkLcOpenSource')


def set_lc_state(mmc, state: str):
    """
    sets the state based on previously defined values
    :param state: str
        from config preset:
        'State0', 'State1', 'State2' ....
    :return: None
    """
    try:
        mmc.setConfig('Channel', state)
    except ValueError:
        print('ERROR: No MicroManager Config Group/Preset named Channel')
    time.sleep(20/1000)