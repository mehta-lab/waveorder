#!/usr/bin/env python
# coding: utf-8

# # 3D phase reconstruction from BF and fluorescence deconvolution


# In[1] Imports and paths
import numpy as np
import matplotlib.pyplot as plt
import os, shutil

import waveorder as wo
from waveorder.io.writer import WaveorderWriter
from waveorder.io.reader import WaveorderReader

plt.style.use(['dark_background'])  # Plotting option for dark background
CompMicro = '/hpc/projects/CompMicro/'
input_file = CompMicro + 'rawdata/hummingbird/Janie/2022_03_10_orgs_nuc_mem_63x_04NA/1_12_pos_1.zarr'
output_path = CompMicro + 'projects/waveorderData/data_processing/2022_03_10_orgs_nuc_mem_63x_04NA/'  # output image path
output_file = 'test_recon_HPC_Shalin_PhaseFluor_JanieData.zarr'

# Delete the test zarr output if it already exists.
if os.path.exists(output_path + output_file):
    shutil.rmtree(output_path + output_file)

## zarr Reader and writer
reader = WaveorderReader(input_file, 'zarr')
writer = WaveorderWriter(output_path, hcs=False, hcs_meta=None, verbose=True)
writer.create_zarr_root(output_file)

## Experimental parameters

N_defocus = reader.mm_meta['IntendedDimensions']['z']
N = reader.mm_meta['Height']
M = reader.mm_meta['Width']
# N_position = reader.mm_meta['IntendedDimensions']['position']
N_position = 76
# N_channel  = reader.mm_meta['IntendedDimensions']['channel']
N_channel = 1
N_time = reader.mm_meta['IntendedDimensions']['time']

# fluorescence parameter
ps_f = 6.5 / 63  # effective pixel size at the sample plane (cam pix/mag in um)
psz = reader.mm_meta['z-step_um']  # z-step size of the defocused intensity stack (in um)
n_media = 1.47  # refractive index of the immersed media for objective (oil: 1.512, water: 1.33, air: 1)
NA_obj = 1.3  # detection NA of the objective
fluor_channel_idx = 3 # 1 = GFP, 2 = mIFP (Histone), 3 = mScarlet (membrane)
lambda_emiss = [0.56, 0.65]  # emission wavelength of the fluorescence channel (list, in um)


# phase recon parameter
lambda_illu = 0.532  # illumination wavelength (um)
NA_illu = 0.4  # illumination NA of the condenser
z_defocus = -(np.r_[:N_defocus] - 0) * psz  # z positions of the stack
BF_channel_idx = 0

# deconvolution parameters
pad_z = 5
use_gpu = False
gpu_id = 0

##  Create Zarr store for given position

chunk_size = (1, 1, 1, N, M)
chan_names = ['Phase3D', 'Membrane-mScarlet']
clims = [(-0.06, 0.06), (500,1E11)] # Estimated contrast limits of channels.

dtype = 'float32'

data_shape = (N_time, len(chan_names), N_defocus, N, M)

writer.init_array(N_position, data_shape, chunk_size, chan_names, dtype, clims,
                  position_name=reader.stage_positions[N_position]['Label'], overwrite=False)
zarr_input = reader.get_zarr(position=N_position) # This array can be loaded lazily.


# In[2]: Phase deconvolution.


## Setup
phase_setup = wo.waveorder_microscopy((N, M), lambda_illu, ps_f, NA_obj, NA_illu, z_defocus, \
                                      n_media=n_media, cali=False, bg_option='global', \
                                      chi=np.pi / 2, \
                                      phase_deconv='3D', pad_z=pad_z, \
                                      use_gpu=use_gpu, gpu_id=gpu_id)


## Iterate through all the time points.
for t in range(N_time):
    phase_result = phase_setup.Phase_recon_3D(
        np.transpose(zarr_input[t, BF_channel_idx], (1, 2, 0)).astype('float32'),
        method='Tikhonov', reg_re=1e-03, autotune_re=False)
    # TODO: when using GPU, we run into out-of-memory error while deconvolving single volume by this line. Need to debug.

    # ## Compare images.
    # plt.figure(figsize=(10, 10))
    # plt.imshow(zarr_input[t, BF_channel_idx, 45], cmap='gray', vmin=0, vmax=1e5)
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(phase_result[:, :, 45], cmap='gray', vmin=clims[0][0], vmax=clims[0][1])
    # plt.colorbar()
    # plt.show()

    writer.write(phase_result.transpose(2, 0, 1), p=N_position, t=t, c=chan_names.index('Phase3D'))


# In[3]: fluorescence deconvolution
fluor_setup = wo.fluorescence_microscopy((N, M, N_defocus), lambda_emiss, ps_f, psz, NA_obj,
                                         n_media=n_media, deconv_mode='3D-WF', pad_z=pad_z, use_gpu=use_gpu,
                                         gpu_id=gpu_id)
bg=0

## Iterate through all time-points


for t in range(N_time):
    # Here you can set the regularization parameters for fluorescence / phase deconvolution
    fluor_result = fluor_setup.deconvolve_fluor_3D(
        np.transpose(zarr_input[t, fluor_channel_idx], (1, 2, 0)).astype('float32'),
        bg, reg=[1e-4],autotune=False)

    # ## Compare images.
    plt.figure(figsize=(10, 10))
    plt.imshow(zarr_input[t, fluor_channel_idx, 45], cmap='gray', vmin=0, vmax=1e5)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(fluor_result[:, :, 45], cmap='gray', vmin=clims[0][0], vmax=clims[0][1])
    plt.colorbar()
    plt.show()

    writer.write(fluor_result.transpose(2, 0, 1), p=N_position, t=t, c=chan_names.index('Membrane-mScarlet'))
    # Initialize the writer for the current position


