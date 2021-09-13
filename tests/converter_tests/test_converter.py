import pytest
import os
import shutil
from recOrder.io.zarr_converter import ZarrConverter

def test_converter():

    input = '/Users/cameron.foltz/Desktop/2021_06_11_recOrder_pytest_20x_04NA 2/2T_3P_81Z_231Y_498X_Kazansky_2'

    output = '/Users/cameron.foltz/Desktop/Test_Data/converer_test/2T_3P_81Z_231Y_498X_Kazansky.zarr'
    if os.path.exists(output):
        shutil.rmtree(output)
    converter = ZarrConverter(input, output)
    converter.run_conversion()
