#!/usr/bin/env python
# coding: utf-8
#
# Example script for simple cpu parallelization with multiprocessing
#
# Set up the job:
# 1) Modify N_processes to decide how many processes you would like to run (up to 256 on bruno).
# 2) If there are more positions than number of processes, they will be split evenly across the processes
#
# A typical workflow looks like:
# >> ssh <user.name>@login01.czbiohub.org
# >> cd /location/of/this/script/
# >> srun --partition=cpu --cpus-per-task=<ENTER-N_processes-here> --mem-per-cpu=<amountG> <path to script>
# >> srun --partition=cpu --cpus-per-task=24 --mem-per-cpu=128G ./Phase3D_fluor_deconv_waveorder_all_channels.py
#
# Prepend the srun command with 'time' to check on your speed savings.
# You can monitor the number of cpus and memory being used with the 'top' command.

import numpy as np
import multiprocessing as mp
import subprocess
from tqdm import tqdm

import waveorder as wo
from waveorder.io.writer import WaveorderWriter
from waveorder.io.reader import WaveorderReader

# Choose number of CPUs to use (must have at least this many available).
N_processes = 24

# Data paths
data_path = '/hpc/projects/comp_micro/rawdata/hummingbird/Janie/2022_03_10_orgs_nuc_mem_63x_04NA/1_12_pos_1.zarr'
output_path = '/hpc/projects/comp_micro/projects/HEK/2022_03_10_orgs_nuc_mem_63x_04NA/multiprocessing/'
output_filename = 'output_test.zarr'
# Channel output names
# regardless of measurement order the phase is always saved at first position, followed by the fluorescence channels!
chan_names = ['Phase3D', 'Deconvolved_GFP', 'Deconvolved_nuc', 'Deconvolved_mem']
dtype = 'float32'

# Experimental parameters
reader = WaveorderReader(data_path, 'zarr')
N_defocus = reader.mm_meta['IntendedDimensions']['z']
N = reader.mm_meta['Height']
M = reader.mm_meta['Width']
# import pdb; pdb.set_trace()
N_position = reader.mm_meta['IntendedDimensions']['position']
N_channel = reader.mm_meta['IntendedDimensions']['channel']
N_time = reader.mm_meta['IntendedDimensions']['time']

# Fluorescence parameter
ps_f = 6.5/63  # effective pixel size at the sample plane (cam pix/mag in um)
psz = reader.mm_meta['z-step_um']  # z-step size of the defocused intensity stack (in um)
n_media = 1.47  # refractive index of the immersed media for objective (oil: 1.512, water: 1.33, air: 1)
lambda_emiss = [0.532, 0.704, 0.594]  # emission wavelength of the fluorescence channel (list, in um)
NA_obj = 1.3  # detection NA of the objective
fluor_channel_idx = [1, 2, 3]
flu_reg = [1e-1, 1e-1, 1e-1]  # define regularization per fluorescence channel

# Phase recon parameter
lambda_illu = 0.532  # illumination wavelength (um)
NA_illu = 0.4  # illumination NA of the condenser
z_defocus = -(np.r_[:N_defocus]-0)*psz  # z positions of the stack
BF_channel_idx = 0  # index of bright field channel
use_gpu = False  # weather or not to use the GPU
gpu_id = 0  # GPU id if use_gpu True
phase_reg = 1e-1  # define regularization for phase reconstruction
pad_z = 5  # z-padding is used for fluorescence deconvolution and phase reconstruction
bg = [50, 50, 50, 50]  # define background values for all channels (labelfree + fluorescence)


# Debugging & profiling:
# The current script runs a full job. Modify or comment the lines that define N, M and N_positions
# to run a short job (~20 seconds) for testing purposes.
# N_position = 24 # Optional truncate number of positions for testing
# N, M = 10, 10 # Optional truncate spatial dimensions for testing

# Kludge! Remove output file manually. TODO: Fix/debug waveorder's writer overwrite.
subprocess.call(['rm', '-r', output_filename])

if N_processes > N_position:
    print("Warning: number of positions should be larger than number of processes.")
if N_processes > mp.cpu_count() :
    print("Warning: number of available cpus should be larger than the number of processes.")

# Waveorder initialization
print('Initializing waveorder ...')
setup = wo.waveorder_microscopy((N, M), lambda_illu, ps_f, NA_obj,
                                NA_illu, z_defocus, n_media=n_media,
                                cali=False, bg_option='global',
                                chi=np.pi/2, phase_deconv='3D',
                                pad_z=pad_z, use_gpu=use_gpu,
                                gpu_id=gpu_id)
fluor_setup = wo.fluorescence_microscopy((N, M, N_defocus),
                                         lambda_emiss, ps_f, psz,
                                         NA_obj, n_media=n_media,
                                         deconv_mode='3D-WF',
                                         pad_z=pad_z, use_gpu=use_gpu,
                                         gpu_id=gpu_id)

# Create Zarr store
print('Creating zarr store ...')
writer = WaveorderWriter(output_path, hcs=False, hcs_meta=None,
                         verbose=True)
writer.create_zarr_root(output_filename)

data_shape = (N_time, N_channel, N_defocus, N, M)
chunk_size = (1,1,1,N,M)

# This function is parallelized over cpus
print('Starting reconstruction ...')
def process_on_single_cpu(pos_start, pos_end):
    for pos in range(pos_start, pos_end+1):
        np_array = reader.get_array(position=pos)
        np_array = np_array[...,0:M,0:N]

        fluor_result = []
        phase_result = []
        for t in range(N_time):
            fluor_result.append(fluor_setup.deconvolve_fluor_3D(np.transpose(np_array[t,fluor_channel_idx],(0,2,3,1)).astype('float32'), bg, reg=flu_reg, autotune=False))
            phase_result.append(setup.Phase_recon_3D(np.transpose(np_array[t,BF_channel_idx],(1,2,0)).astype('float32'), method='Tikhonov', reg_re=phase_reg, autotune_re=False))

        fluor_result = np.array(fluor_result)
        zarr_data_array = []
        zarr_data_array.append(np.array(phase_result))
        clims = [(-0.06,0.06)]
        if len(fluor_channel_idx) == 1:
            zarr_data_array.append(fluor_result)
            clims.append((0, np.max(fluor_result)/2))
        else:
            for fl in range(len(fluor_channel_idx)):
                zarr_data_array.append(fluor_result[:,fl])
                clims.append((0, np.max(fluor_result[:,fl])/2))

        writer.init_array(pos, data_shape, chunk_size, chan_names, dtype, clims, position_name=reader.stage_positions[pos]['Label'], overwrite=True)
        zarr_data_array = np.transpose(np.array(zarr_data_array),(1,0,4,2,3))
        writer.write(zarr_data_array, p=pos)


if __name__ == '__main__':
    # Set up processes
    processes = []
    pos_per_process = int(np.ceil(N_position/N_processes))
    for i in range(N_processes):
        pos_start = i*pos_per_process
        pos_end = np.minimum(pos_start + pos_per_process - 1, N_position)
        print("Preparing to run positions "+str(pos_start)+'-'+str(pos_end)+' on process '+str(i))
        processes.append(mp.Process(target=process_on_single_cpu, args=(pos_start, pos_end,)))

    # Run processes
    for p in processes:
        p.start()
    for p in tqdm(processes):
        p.join()
