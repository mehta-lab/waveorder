from unittest.mock import MagicMock, Mock, call
import pytest
import numpy as np
from numpy import ndarray
from typing import Callable, Tuple

# tested components
from recOrder.io.core_functions import *


# TODO: move these to fixture or generate with Hypothesis
# dynamic range
TIFF_I_MAX = 2**16
# image size
IMAGE_WIDTH = np.random.randint(1, 2**12)
IMAGE_HEIGHT = np.random.randint(1, 2**12)
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH
# serialized image from the pycromanager bridge
SERIAL_IMAGE = np.random.randint(0, TIFF_I_MAX, size=(PIXEL_COUNT,))
# LC device parameters
# TODO: parameterize this example
DEVICE_PROPERTY = ("deviceName", "propertyName")
CONFIG_GROUP = "configGroup"
CONFIG_NAME = "State0"
# LC state in native units
LC_STATE = np.random.rand(1)[0] * 10


def _get_mmcore_mock():
    """Creates a mock for the `pycromanager.Core` object.

    Returns
    -------
    MagicMock
        MMCore mock object
    """
    mmcore_mock_config = {
        "getImage": Mock(return_value=SERIAL_IMAGE),
        "getProperty": Mock(return_value=str(LC_STATE)),
    }
    return MagicMock(**mmcore_mock_config)


def _get_snap_manager_mock():
    """Creates a mock for the pycromanager remote Snap Live Window Manager object.

    Returns
    -------
    MagicMock
        Mock object for `org.micromanager.internal.SnapLiveManager` via pycromanager
    """
    sm = MagicMock()
    get_snap_mocks = {
        "getHeight": Mock(return_value=IMAGE_HEIGHT),
        "getWidth": Mock(return_value=IMAGE_WIDTH),
        "getRawPixels": Mock(return_value=SERIAL_IMAGE),
    }
    # TODO: break down these JAVA call stack chains for maintainability
    sm.getDisplay.return_value.getDisplayedImages.return_value.get = Mock(
        # return image object mock with H, W, and pixel values
        return_value=Mock(**get_snap_mocks)
    )
    sm.getDisplay.return_value.getImagePlus.return_value.getStatistics = Mock(
        # return statistics object mock with the attribute "umean"
        return_value=Mock(umean=SERIAL_IMAGE.mean())
    )
    return sm


def _is_int(data: ndarray):
    """Check if the data type is integer.

    Parameters
    ----------
    data

    Returns
    -------
    bool
        True if the data type is any integer type.
    """
    return np.issubdtype(data.dtype, np.integer)


def _get_examples(low: float, high: float):
    """Generate 4 valid and 4 invalid floating numbers for closed interval [low, high].

    Parameters
    ----------
    low : float
    high : float

    Returns
    -------
    tuple(1d-array, 1d-array)
        valid and invalid values
    """
    epsilon = np.finfo(float).eps
    samples = np.random.rand(4)
    valid_values = samples * (high - low) + low + epsilon
    invalid_values = np.array(
        [
            low - samples[0],
            low - samples[1],
            high + samples[2],
            high + samples[3],
        ]
    )
    return valid_values, invalid_values


def test_suspend_live_sm():
    """Test `recOrder.io.core_functions.suspend_live_sm`."""
    snap_manager = _get_snap_manager_mock()
    with suspend_live_sm(snap_manager) as sm:
        sm.setSuspended.assert_called_once_with(True)
    snap_manager.setSuspended.assert_called_with(False)


def test_snap_and_get_image():
    """Test `recOrder.io.core_functions.snap_and_get_image`."""
    sm = _get_snap_manager_mock()
    image = snap_and_get_image(sm)
    assert _is_int(image), image.dtype
    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH), image.shape


def test_snap_and_average():
    """Test `recOrder.io.core_functions.snap_and_average`."""
    sm = _get_snap_manager_mock()
    mean = snap_and_average(sm)
    np.testing.assert_almost_equal(mean, SERIAL_IMAGE.mean())


def _set_lc_test(
    tested_func: Callable[[object, Tuple[str, str], float], None],
    value_range: Tuple[float, float],
):
    mmc = _get_mmcore_mock()
    valid_values, invalid_values = _get_examples(*value_range)
    for value in valid_values:
        tested_func(mmc, DEVICE_PROPERTY, value)
        mmc.setProperty.assert_called_with(
            DEVICE_PROPERTY[0], DEVICE_PROPERTY[1], str(value)
        )
    for value in invalid_values:
        with pytest.raises(ValueError):
            tested_func(mmc, DEVICE_PROPERTY, value)


def test_set_lc_waves():
    """Test `recOrder.io.core_functions.set_lc_waves`."""
    _set_lc_test(set_lc_waves, (0.001, 1.6))


def test_set_lc_voltage():
    """Test `recOrder.io.core_functions.set_lc_voltage`."""
    _set_lc_test(set_lc_voltage, (0.0, 20.0))


def test_set_lc_daq():
    """Test `recOrder.io.core_functions.set_lc_daq`."""
    _set_lc_test(set_lc_daq, (0.0, 5.0))


def test_get_lc():
    """Test `recOrder.io.core_functions.get_lc`."""
    mmc = _get_mmcore_mock()
    state = get_lc(mmc, DEVICE_PROPERTY)
    mmc.getProperty.assert_called_once_with(*DEVICE_PROPERTY)
    np.testing.assert_almost_equal(state, LC_STATE)


def test_define_meadowlark_state():
    """Test `recOrder.io.core_functions.define_meadowlark_state`."""
    mmc = _get_mmcore_mock()
    define_meadowlark_state(mmc, DEVICE_PROPERTY)
    mmc.setProperty.assert_called_once_with(*DEVICE_PROPERTY, 0)
    mmc.waitForDevice.assert_called_once_with(DEVICE_PROPERTY[0])


def test_define_config_state():
    """Test `recOrder.io.core_functions.define_config_state`."""
    mmc = _get_mmcore_mock()
    device_properties = [DEVICE_PROPERTY] * 4
    values = _get_examples(0, 10)[0].tolist()
    define_config_state(
        mmc, CONFIG_GROUP, CONFIG_NAME, device_properties, values
    )
    expected_calls = [
        call(CONFIG_GROUP, CONFIG_NAME, *d, str(v))
        for d, v in zip(device_properties, values)
    ]
    got_calls = mmc.defineConfig.call_args_list
    assert got_calls == expected_calls, got_calls


def test_set_lc_state():
    """Test `recOrder.io.core_functions.set_lc_state`."""
    mmc = _get_mmcore_mock()
    set_lc_state(mmc, CONFIG_GROUP, CONFIG_NAME)
    mmc.setConfig.assert_called_once_with(CONFIG_GROUP, CONFIG_NAME)
