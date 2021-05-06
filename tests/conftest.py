import pytest
import shutil
import os

@pytest.fixture(scope='function')
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

@pytest.fixture(scope="session")
def setup_mm2gamma_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1UWSr4GQ6Kpj5irq2TicvDLULfWjKhh0b/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_ometiffs = '1UWSr4GQ6Kpj5irq2TicvDLULfWjKhh0b'

    # download files to temp folder
    output = temp_gamma + "/mm2gamma_ometiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_ometiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_gamma, 'ome-tiffs')
    subfolders = [f for f in os.listdir(src) if os.path.isdir(join(src, f))]

    # specific folder
    one_folder = join(src, subfolders[0])
    # random folder
    rand_folder = join(src, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield src, one_folder, rand_folder

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_gamma, 'ome-tiffs'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")
