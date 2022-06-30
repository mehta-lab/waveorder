import pytest
import shutil
import os
from wget import download

@pytest.fixture(scope="session")
def setup_test_data():
    # create /pytest_temp/ and /pytest_temp/rawdata/ folders,
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_data = os.path.join(temp_folder, 'rawdata')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(temp_data):
        os.mkdir(temp_data)

    #  download data to /pytest_temp/rawdata/recOrder/ folder if it doesn't already exist
    url = 'https://zenodo.org/record/6249285/files/recOrder_testData.zip?download=1'
    output = temp_data + "/recOrder_testData.zip"
    if not os.path.isdir(temp_data+'/recOrder/'):
        print("Downloading test files...")
        download(url, out=output)
        shutil.unpack_archive(output, extract_dir=temp_data)

    ometiff_data = os.path.join(temp_data,
                                'recOrder/2021_06_11_recOrder_pytest_20x_04NA/2T_3P_81Z_231Y_498X_Kazansky_2')
    zarr_data = os.path.join(temp_data,
                             'recOrder/2021_06_11_recOrder_pytest_20x_04NA_zarr/2T_3P_81Z_231Y_498X_Kazansky.zarr')
    bf_data = os.path.join(temp_data,
                           'recOrder/2021_06_11_recOrder_pytest_20x_04NA_BF_zarr/2T_3P_81Z_231Y_498X_Kazansky.zarr')

    yield temp_data, ometiff_data, zarr_data, bf_data

# create /pytest_temp/data_save folder for each test then delete when test is done
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