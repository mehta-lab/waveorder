import pytest
from deconorder.io.config_reader import ConfigReader


def test_config_reader_change():
    with ConfigReader('/Users/cameron.foltz/decOrder/deconorder/examples/config_example.yml') as config:
        config.data_dir = 'test'

def test_config_reader_load():
    with ConfigReader('/Users/cameron.foltz/decOrder/deconorder/examples/config_example.yml') as config:
        print(config.data_dir)