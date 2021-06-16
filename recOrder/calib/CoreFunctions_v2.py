import time
import numpy as np

def snap_image(mmc):

    #TODO: calculate mean in java before returning?
    mmc.snapImage()

    return mmc.getImage()

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
    set_lc(mmc, lca, PROPERTIES['LCB'])

    print("setting LCA = "+str(lca))
    print("setting LCB = "+str(lcb))
    print("\n")

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