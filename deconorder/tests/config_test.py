# import pytest
from deconorder.io.config_reader import ConfigReader


def test_config_reader_change():

    config = ConfigReader('/Users/cameron.foltz/decOrder/deconorder/examples/config_example.yml')
    print(config.data_dir)
    config.data_dir='test'
    print(config.data_dir)

def test_config_reader_load():
    with ConfigReader('/Users/cameron.foltz/decOrder/deconorder/examples/config_example.yml') as config:
        print(config.z_slices)

