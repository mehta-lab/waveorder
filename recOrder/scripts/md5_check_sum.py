import hashlib
import zarr
import os
import numpy as np
import pathlib as path
import click

@click.command()
@click.option('--zarr_path', required=True, type=str, help='path to the save directory')
@click.option('--raw_stats_path', required=True, type=str, help='path to the statistics file of the raw data')
def parse_args(zarr_path, raw_stats_path):
    """parse command line arguments and return class with the arguments"""

    class Args():
        def __init__(self, zarr_path, raw_stats_path):
            self.zarr_path = zarr_path
            self.raw_stats_path = raw_stats_path

    return Args(zarr_path, raw_stats_path)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def gen_stats_file(zarr_path, save_path):

    name = path.PurePath(zarr_path).name
    if name.endswith('.zarr'):
        name = name[:-5]
    file_name = os.path.join(save_path, name+'_ZarrStatistics.txt')
    file = open(file_name, 'w')

    zstore = zarr.open(zarr_path, 'r')['array']
    shape = zstore.shape

    for p in range(shape[0]):
        for t in range(shape[1]):
            for c in range(shape[2]):
                for z in range(shape[3]):

                    image = zstore[p, t, c, z]
                    mean = np.mean(image)
                    median = np.median(image)
                    std = np.std(image)
                    file.write(f'Coord: {(p, t, c, z)}, Mean: {mean}, Median: {median}, Std: {std}\n')

    file.close()
    return file_name


if __name__ == "__main__":

    Args = parse_args(standalone_mode=False)

    save_path = str(path.PurePath(Args.raw_stats_path).parent)
    conv_stats_path = gen_stats_file(Args.zarr_path, save_path)

    raw_md5 = md5(Args.raw_stats_path)
    converted_md5 = md5(conv_stats_path)

    if raw_md5 != converted_md5:
        print('MD5 check sum failed.  Potential Error in Conversion')
    else:
        print('MD5 check sum passed. Conversion successful')





