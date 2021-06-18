import pytest
import shutil
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

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
    temp_pipeline = os.path.join(temp_folder, 'pipeline_test')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1UWSr4GQ6Kpj5irq2TicvDLULfWjKhh0b/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    recOrder_pytest = '1_FoKVyl4Qa4F-4_vaREq_fqxODEU5neU'

    # download files to temp folder
    output = temp_pipeline + "/recOrder_pytest.zip"
    gdd.download_file_from_google_drive(file_id=recOrder_pytest,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_pipeline, '2021_06_11_recOrder_pytest_20x_04NA')
    data = os.path.join(src, '2T_3P_81Z_231Y_498X_Kazansky_2')

    yield temp_pipeline, data

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_pipeline, '2021_06_11_recOrder_pytest_20x_04NA'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")
