import pytest
from ..conftest import setup_folder_qlipp_pipeline
from recOrder.io.config_reader import ConfigReader
# from recOrder.pipelines.run_pipeline import run_pipeline


def test_pipeline_working():
    config = ConfigReader('../../recOrder/examples/config_QLIPP_3D.yml')
    run_pipeline(config)

    pass

def test_pipeline_quality():
    pass