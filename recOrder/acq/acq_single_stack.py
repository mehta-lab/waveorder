import numpy as np
from recOrder.calib.CoreFunctions import set_lc_state, snap_and_get_image

def acquire_2D(mm, mmc, scheme, snap_manager=None):

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

    if not snap_manager:
        snap_manager = mm.getSnapLiveManager()

    stage = mmc.getFocusDevice()
    current_z = mmc.getPosition(stage)
    n_channels = 4 if scheme == '4-State' else 5
    stack = []
    for c in range(n_channels):
        set_lc_state(mmc, f'State{c}')
        z_stack = []

        for z in range(current_z-z_start, current_z+z_end+z_step, z_step):
            mmc.setPosition(stage, z)
            z_stack.append(snap_and_get_image(snap_manager))

        stack.append(z_stack)

    return np.asarray(stack)
