# This script can be modified to debug and test calibrations

import napari
import time, random
from contextlib import contextmanager
from pycromanager import Core
from recOrder.plugin.main_widget import MainWidget

SAVE_DIR = "."
SWING = 0.05
CAL_REPEATS = 3
BKG_REPEATS = 3


@contextmanager
def stage_detour(app: MainWidget, dx: float, dy: float, wait=5):
    """Context manager to temporarily move the stage to a new XY-position.

    Parameters
    ----------
    app : MainWidget
        recOrder main widget instance
    dx : float
        relative x to translate
    dy : float
        relative y to translate
    wait : int, optional
        time to wait for the stage to complete movement, by default 5

    Yields
    ------
    MainWidget
        recOrder main widget instance

    Usage
    -----
    ```py
    with stage_detour(app) as app:
        pass # do something at the new location
    ```
    """
    xy_stage = app.mmc.getXYStageDevice()
    # get the original position
    ox = app.mmc.getXPosition(xy_stage)
    oy = app.mmc.getYPosition(xy_stage)
    # go to a translated position
    # TODO: args are floored due to a pycromanager bug: https://github.com/micro-manager/pycro-manager/issues/67
    app.mmc.setRelativeXYPosition(int(dx), int(dy))
    time.sleep(wait)
    try:
        yield app
    finally:
        # go back to the original position
        # TODO: args are floored due to a pycromanager bug: https://github.com/micro-manager/pycro-manager/issues/67
        app.mmc.setXYPosition(int(ox), int(oy))
        time.sleep(wait)


def measure_fov(mmc: Core):
    """Calculate the MM FOV in microns.

    Parameters
    ----------
    mmc : Core
        MMCore object via pycromanager (with CamelCase set to `True`)

    Returns
    -------
    tuple[float, float]
        FOV size (x, y)
    """
    pixel_size = float(mmc.getPixelSizeUm())
    if pixel_size == 0:
        float(
            input(
                "Pixel size is not calibrated. Please provide an estimate (in microns):"
            )
        )
    fov_x = pixel_size * float(mmc.getImageWidth())
    fov_y = pixel_size * float(mmc.getImageHeight())
    return fov_x, fov_y


def rand_shift(length: float):
    """Randomly signed shift of a certain length.

    Parameters
    ----------
    length : float
        absolote length in microns

    Returns
    -------
    float
        +length or -length
    """
    sign = random.randint(0, 1) * 2 - 1
    return sign * length


def main():
    viewer = napari.Viewer()
    app = MainWidget(viewer)
    viewer.window.add_dock_widget(app)
    app.ui.qbutton_gui_mode.click()
    app.calib_scheme = "5-State"
    app.directory = SAVE_DIR
    app.save_directory = SAVE_DIR

    fov_x, fov_y = measure_fov(app.mmc)

    input("Please center the target in the FOV and hit <Enter>")

    for cal_repeat in range(CAL_REPEATS):
        dx = rand_shift(fov_x)
        dy = rand_shift(fov_y)
        # run calibration
        with stage_detour(app, dx, dy) as app:
            print(f"Calibration repeat # {cal_repeat}")
            app.swing = SWING

            print(f"Calibrating with swing = {SWING}")
            app.run_calibration()
            time.sleep(90)

        for bkg_repeat in range(BKG_REPEATS):
            # capture background
            with stage_detour(app, dx, dy) as app:
                print(f">>> Background repeat # {bkg_repeat}")
                app.last_calib_meta_file = app.calib.meta_file
                app.capture_bg()
                time.sleep(20)
            app.ui.cb_bg_method.setCurrentIndex(
                1
            )  # Set to "Measured" bg correction
            app.enter_bg_correction()
            app.save_name = f"cal-{cal_repeat}-bkg-{bkg_repeat}"
            app.enter_acq_bg_path()
            app.acq_birefringence()
            time.sleep(15)


if __name__ == "__main__":
    main()
