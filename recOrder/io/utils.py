import glob
import os

def gather_sub_dir(self, dir_):
    files = glob.glob(dir_ + '*')

    dirs = []
    for direc in files:
        if os.path.isdir(direc):
            dirs.append(direc)

    return dirs