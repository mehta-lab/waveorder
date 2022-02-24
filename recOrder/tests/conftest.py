import pytest
import shutil
import os
from wget import download

@pytest.fixture(scope='session')
def setup_folder_qlipp_pipeline():
    temp_folder = os.getcwd() + '/pytest_temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    yield temp_folder

    try:
        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope='function')
def setup_data_save_folder():
    temp_folder = os.getcwd() + '/pytest_temp'
    data_save_folder = os.path.join(temp_folder, 'data_save')
    if not os.path.isdir(data_save_folder):
        os.mkdir(data_save_folder)
        print("\nsetting up data_save folder")

    yield data_save_folder

    try:
        # remove temp folder
        shutil.rmtree(data_save_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope="session")
def setup_test_data():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_data = os.path.join(temp_folder, 'rawdata')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(temp_data):
        os.mkdir(temp_data)

    # Zenodo URL
    url = 'https://zenodo.org/record/6249285/files/recOrder_testData.zip?download=1'

    # download files to temp folder
    output = temp_data + "/recOrder_testData.zip"
    download(url, out=output)
    shutil.unpack_archive(output, extract_dir=temp_data)

    ometiff_data = os.path.join(temp_data,
                                'recOrder/2021_06_11_recOrder_pytest_20x_04NA/2T_3P_81Z_231Y_498X_Kazansky_2')
    zarr_data = os.path.join(temp_data,
                             'recOrder/2021_06_11_recOrder_pytest_20x_04NA_zarr/2T_3P_81Z_231Y_498X_Kazansky.zarr')
    bf_data = os.path.join(temp_data,
                           'recOrder/2021_06_11_recOrder_pytest_20x_04NA_BF_zarr/2T_3P_81Z_231Y_498X_Kazansky.zarr')

    yield temp_data, ometiff_data, zarr_data, bf_data

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(temp_data)

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")
