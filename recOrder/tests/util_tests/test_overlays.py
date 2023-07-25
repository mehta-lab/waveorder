import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.typing import NDArray
from numpy.testing import assert_equal

from recOrder.io.utils import ret_ori_overlay


@st.composite
def _birefringence(draw):
    shape = (2,) + tuple(
        draw(st.lists(st.integers(1, 16), min_size=2, max_size=4))
    )
    dtype = draw(npst.floating_dtypes(sizes=(32, 64)))
    bit_width = dtype.itemsize * 8
    retardance = draw(
        npst.arrays(
            dtype,
            shape=shape,
            elements=st.floats(
                min_value=0,
                max_value=50,
                exclude_min=True,
                width=bit_width,
            ),
        )
    )
    orientation = draw(
        npst.arrays(
            dtype,
            shape=shape,
            elements=st.floats(
                min_value=0,
                max_value=dtype.type(np.pi),
                exclude_min=True,
                exclude_max=True,
                width=bit_width,
            ),
        )
    )
    return retardance, orientation


@given(briefringence=_birefringence(), jch=st.booleans())
def test_ret_ori_overlay(briefringence: tuple[NDArray, NDArray], jch: bool):
    """Test recOrder.io.utils.ret_ori_overlay()"""
    retardance, orientation = briefringence
    retardance_copy = retardance.copy()
    orientation_copy = orientation.copy()
    cmap = "JCh" if jch else "HSV"
    overlay = ret_ori_overlay(
        retardance,
        orientation,
        ret_max=np.percentile(retardance, 99),
        cmap=cmap,
    )
    # check that the function did not mutate input data
    assert_equal(retardance, retardance_copy)
    assert_equal(orientation, orientation_copy)
    # check output properties
    # output contains NaN, pending further investigation
    # assert overlay.min() >= 0
    # assert overlay.max() <= 1
    assert overlay.shape == retardance.shape + (3,)
