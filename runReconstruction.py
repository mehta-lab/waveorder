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
import argparse
import shutil
import os
from recOrder.pipelines.run_pipeline import run_pipeline
from recOrder.io.config_reader import ConfigReader


# @click.command()
# @click.option('--config', required=True, type=str, help='path to config yml file')
# def parse_args(config):
#     """Parse command line arguments
#     In python namespaces are implemented as dictionaries
#     :return: namespace containing the arguments passed.
#     """
#
#     if os.path.exists(config):
#         return config
#
#     else:
#         raise ValueError('Specified path does not exist')


def parse_args():
    """Parse command line arguments
    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args):
        raise ValueError('Specified path does not exist')

    config = ConfigReader(args)
    run_pipeline(config)
    shutil.copy(args, os.path.join(config.processed_dir, 'config.yml'))
