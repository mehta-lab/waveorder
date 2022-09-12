import numpy as np
import json
import os
from recOrder.io.core_functions import set_lc_state, snap_and_get_image
from waveorder.io.reader import WaveorderReader
import time
import glob

def generate_acq_settings(mm, channel_group, channels=None, zstart=None, zend=None, zstep=None,
                          save_dir=None, prefix=None, keep_shutter_open_channels=False, keep_shutter_open_slices=False):
    """
    This function generates a json file specific to the micromanager SequenceSettings.
    It has default parameters for a multi-channels z-stack acquisition but does not yet
    support multi-position or multi-frame acquisitions.

    This also has default values for QLIPP Acquisition.  Can be used as a framework for other types
    of acquisitions

    Parameters
    ----------
    mm:             (object) MM Studio API object
    scheme:         (str) '4-State' or '5-State'
    zstart:         (float) relative starting position for the z-stack
    zend:           (float) relative ending position for the z-stack
    zstep:          (float) step size for the z-stack
    save_dir:       (str) path to save directory
    prefix:         (str) name to save the data under

    Returns
    -------
    settings:       (json) json dictionary conforming to MM SequenceSettings
    """

    # Get API Objects
    am = mm.getAcquisitionManager()
    ss = am.getAcquisitionSettings()
    app = mm.app()

    # Get current SequenceSettings to modify
    original_ss = ss.toJSONStream(ss)
    original_json = json.loads(original_ss).copy()

    if zstart:
        do_z = True
    else:
        do_z = False

    # Structure of the channel properties
    channel_dict = {'channelGroup': channel_group,
                    'config': None,
                    'exposure': None,
                    'zOffset': 0,
                    'doZStack': do_z,
                    'color': {'value': -16747854, 'falpha': 0.0},
                    'skipFactorFrame': 0,
                    'useChannel': True if channels else False}

    channel_list = None
    if channels:
        # Append all the channels with their current exposure settings
        channel_list = []
        for chan in channels:
            #todo: think about how to deal with missing exposure
            exposure = app.getChannelExposureTime(channel_group, chan, 10) # sets exposure to 10 if not found
            channel = channel_dict.copy()
            channel['config'] = chan
            channel['exposure'] = exposure

            channel_list.append(channel)

    # set other parameters
    original_json['numFrames'] = 1
    original_json['intervalMs'] = 0
    original_json['relativeZSlice'] = True
    original_json['slicesFirst'] = True
    original_json['timeFirst'] = False
    original_json['keepShutterOpenSlices'] = keep_shutter_open_slices
    original_json['keepShutterOpenChannels'] = keep_shutter_open_channels
    original_json['useAutofocus'] = False
    original_json['saveMode'] = 'MULTIPAGE_TIFF'
    original_json['save'] = True if save_dir else False
    original_json['root'] = save_dir if save_dir else ''
    original_json['prefix'] = prefix if prefix else 'Untitled'
    original_json['channels'] = channel_list
    original_json['zReference'] = 0.0
    original_json['channelGroup'] = channel_group
    original_json['usePositionList'] = False
    original_json['shouldDisplayImages'] = True
    original_json['useSlices'] = do_z
    original_json['useFrames'] = False
    original_json['useChannels'] = True if channels else False
    original_json['slices'] = list(np.arange(float(zstart), float(zend+zstep), float(zstep))) if zstart else []
    original_json['sliceZStepUm'] = zstep
    original_json['sliceZBottomUm'] = zstart
    original_json['sliceZTopUm'] = zend
    original_json['acqOrderMode'] = 1

    return original_json

def acquire_from_settings(mm, settings, grab_images = True):
    """
    Function to acquire an MDA acquisition with the native MM MDA Engine.
    Assumes single position acquisition.

    Parameters
    ----------
    mm:             (object) MM Studio API object
    settings:       (json) JSON dictionary conforming to MM SequenceSettings
    grab_images:    (bool) True/False if you want to return the acquired array

    Returns
    -------

    """

    am = mm.getAcquisitionManager()
    ss = am.getAcquisitionSettings()

    ss_new = ss.fromJSONStream(json.dumps(settings))
    am.runAcquisitionWithSettings(ss_new, True)

    time.sleep(3)

    #TODO: speed improvements in reading the data with pycromanager acquisition?
    if grab_images:
        # get the most recent acquisition if multiple
        path = os.path.join(settings['root'], settings['prefix'])
        files = glob.glob(path+'*')
        index = max([int(x.split(path + '_')[1]) for x in files])

        reader = WaveorderReader(path+f'_{index}', 'ometiff', extract_data=True)

        return reader.get_array(0)

def acquire_2D(mm, mmc, scheme, snap_manager=None):
    """
    Acquire a 2D stack with pycromanager (data transfer over ZMQ) given the acquisition scheme.

    Parameters
    ----------
    mm:             (object) MM studio object
    mmc:            (object) MM Core object
    scheme:         (str) '5-state' or '4-State'
    snap_manager:   (object) MM Snap Live Window object

    Returns
    -------
    image_stack:    (nd-array) nd-array of size (4, Y, X) or (5, Y, X)

    """

    if not snap_manager:
        snap_manager = mm.getSnapLiveManager()

    set_lc_state(mmc, 'State0')
    state0 = snap_and_get_image(snap_manager)

    set_lc_state(mmc, 'State1')
    state1 = snap_and_get_image(snap_manager)

    set_lc_state(mmc, 'State2')
    state2 = snap_and_get_image(snap_manager)

    set_lc_state(mmc, 'State3')
    state3 = snap_and_get_image(snap_manager)

    if scheme == '5-State':
        set_lc_state(mmc, 'State4')
        state4 = snap_and_get_image(snap_manager)
        return np.asarray([state0, state1, state2, state3, state4])

    else:
        return np.asarray([state0, state1, state2, state3])


def acquire_3D(mm, mmc, scheme, z_start, z_end, z_step, snap_manager=None):
    """
    Acquire a 3D stack with pycromanager (data transfer over ZMQ) given the acquisition scheme.

    Parameters
    ----------
    mm:             (object) MM studio object
    mmc:            (object) MM Core object
    scheme:         (str) '5-state' or '4-State'
    z_start:        (float) relative z-start location [um]
    z_end:          (float) relative z-end location [um]
    z_step:         (float) z-step [um]
    snap_manager:   (object) MM Snap Live Window object

    Returns
    -------
    image_stack:    (nd-array) nd-array of size (Z, 4, Y, X) or (Z, 5, Y, X)

    """

    if not snap_manager:
        snap_manager = mm.getSnapLiveManager()

    stage = mmc.getFocusDevice()
    current_z = mmc.getPosition(stage)
    n_channels = 4 if scheme == '4-State' else 5
    stack = []
    for c in range(n_channels):
        set_lc_state(mmc, f'State{c}')
        z_stack = []
        for z in np.arange(current_z+z_start, current_z+z_end+z_step, z_step):
            mmc.setPosition(stage, z)
            z_stack.append(snap_and_get_image(snap_manager))

        stack.append(z_stack)

    mmc.setPosition(stage, current_z)

    return np.asarray(stack)
