"""
runReconstruction:
Reconstruct birefringence, slow axis, transmission, and degree of polarization from polarization-resolved images.
This script provides a convenient method to workflow process multi-dimensional images acquired with Micro-Manager and OpenPolScope acquisition plugin.
Parameters
----------
    --config: path to configuration file.
Returns
-------
    None: the script writes data to disk.
"""

import click
import shutil
import os

from recOrder.pipelines.run_pipeline import run_pipeline
from recOrder.io.config_reader import ConfigReader


@click.option('--config', required=True, type=str, help='path to config yml file')
def parse_args(path):
    """Parse command line arguments
    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    if os.path.exists(path):
        return path

    else:
        raise ValueError('Specified path does not exist')

if __name__ == '__main__':
    cfg_path = parse_args()
    config = ConfigReader(cfg_path)
    run_pipeline(config)
    shutil.copy(cfg_path, os.path.join(config.processed_dir, 'config.yml'))
