#!/usr/bin/env python
#
# QLIPP 4 state reconstruction and fluorescence deconvolution
#
# Set up the job:
# 1) Modify N_processes to decide how many processes you would like to run (up to 256 on bruno).
# 2) If there are more positions than number of processes, they will be split evenly across the processes
#
# A typical workflow looks like:
# >> ssh <user.name>@login01.czbiohub.org
# >> cd /location/of/this/script/
# >> srun --partition=cpu --cpus-per-task=<ENTER-N_processes-here> --mem-per-cpu=<amountG> <path to script>
# >> srun --partition=cpu --cpus-per-task=24 --mem-per-cpu=128G ./QLIPP_4_states_fluor_deconv_waveorder.py
#
# Prepend the srun command with 'time' to check on your speed savings.
# You can monitor the number of cpus and memory being used with the 'top' command.


import os
import glob
import json
import tifffile
import multiprocessing as mp
import subprocess
from tqdm import tqdm

import waveorder as wo
from waveorder.io.writer import WaveorderWriter
from waveorder.io.reader import WaveorderReader
from waveorder.background_estimator import *

# Choose number of CPUs to use (must have at least this many available).
N_processes = 10

CompMicro = '/hpc/projects/comp_micro/'  # set CompMicro mount path
data_path = CompMicro+'rawdata/hummingbird/Janie/2022_03_29_GOLGA2_nuc_mem_LF_63x_04NA/test_no_pertubation_mem100.zarr'  # path to zarr
calibration_metadata = CompMicro+'rawdata/hummingbird/Janie/2022_03_29_GOLGA2_nuc_mem_LF_63x_04NA/calibration_metadata.txt'  # path to calibration
bg_path = CompMicro+'rawdata/hummingbird/Janie/2022_03_29_GOLGA2_nuc_mem_LF_63x_04NA/backgrounds/BG_no_pertubation'  # path to background

# read metadata
reader = WaveorderReader(data_path, 'zarr')
N_defocus = reader.mm_meta['IntendedDimensions']['z']
N = reader.mm_meta['Height']
M = reader.mm_meta['Width']
N_position = reader.mm_meta['IntendedDimensions']['position']
N_channel = reader.mm_meta['IntendedDimensions']['channel']
N_time = reader.mm_meta['IntendedDimensions']['time']
filedir_bg = bg_path + '*.tif'
files_bg = sorted(glob.glob(filedir_bg), key=wo.numericalSort)
I_bg = []
for i in range(len(files_bg)):
    I_bg.append(tifffile.imread(files_bg[i]).astype('float32'))
I_bg = np.array(I_bg)
calib_meta = json.load(open(calibration_metadata))

# fluorescence parameter
ps_f = 6.5/63  # effective pixel size at the sample plane (cam pix/mag in um)
psz = reader.mm_meta['z-step_um']  # z-step size of the defocused intensity stack (in um)
n_media = 1.47  # refractive index of the immersed media for objective (oil: 1.512, water: 1.33, air: 1)
lambda_emiss = [0.532, 0.594, 0.704]  # emission wavelength of the fluorescence channel (list, in um)
NA_obj = 1.3  # detection NA of the objective
fluor_channel_idx = [0, 1, 2]  # idx of fluorescence channels
flu_reg = [8e-4, 8e-4, 8e-4]  # regularization for fluorescence deconvolution per channel

# phase recon parameter
lambda_illu = 0.532  # illumination wavelength (um)
NA_illu = 0.4  # illumination NA of the condenser
z_defocus = -(np.r_[:N_defocus]-0)*psz  # z positions of the stack
swing = calib_meta['Summary']['Swing (fraction)']
QLIPP_channel_idx = [3, 4, 5, 6]  # idx of QLIPP channels
phase_reg = 1e-3  # phase regularization

pad_z = 25  # applied for phase reconstruction and deconvolution
use_gpu = False
gpu_id = 0

output_path = CompMicro+'projects/HEK/2022_03_29_GOLGA2_nuc_mem_LF_63x_04NA/'  # output directory
output_name = 'testrun_no_pertubation_mp.zarr'  # name of output zarr store
chan_names = ['Phase3D', 'Retardance', 'Orientation', 'Brightfield', 'Deconvolved_Golgi', 'Deconvolved_Membrane', 'Deconvolved_Nucleus']  # Channel names of output, the first 4 channels are QLIPP


# Kludge! Remove output file manually. TODO: Fix/debug waveorder's writer overwrite.
subprocess.call(['rm', '-r', os.path.join(output_path, output_name)])
# start reconstruction
dtype = 'float32'
writer = WaveorderWriter(output_path, hcs=False, hcs_meta=None, verbose=True)
writer.create_zarr_root(output_name)
data_shape = (N_time, N_channel, N_defocus, N, M)
chunk_size = (1,1,1, N, M)

inst_mat = np.array([[1, 0, 0, -1],
                     [1, np.sin(2 * np.pi * swing), 0, -np.cos(2 * np.pi * swing)],
                     [1, -0.5 * np.sin(2 * np.pi * swing), np.sqrt(3) * np.cos(np.pi * swing) * np.sin(np.pi * swing), -np.cos(2 * np.pi * swing)],
                     [1, -0.5 * np.sin(2 * np.pi * swing), -np.sqrt(3) / 2 * np.sin(2 * np.pi * swing), -np.cos(2 * np.pi * swing)]])

QLIPP_setup = wo.waveorder_microscopy((N,M), lambda_illu, ps_f, NA_obj, NA_illu, z_defocus, n_media=n_media, cali=True, bg_option='local_fit',                                       A_matrix=inst_mat, chi = swing,                                       phase_deconv='3D', illu_mode='BF', pad_z=pad_z,                                      use_gpu=use_gpu, gpu_id=gpu_id)
fluor_setup = wo.fluorescence_microscopy((N, M, N_defocus), lambda_emiss, ps_f, psz, NA_obj, 
                                         n_media=n_media, deconv_mode='3D-WF', pad_z=pad_z, use_gpu=use_gpu, gpu_id=gpu_id)

# This function is parallelized over cpus
print('Starting reconstruction ...')
def process_on_single_cpu(pos_start, pos_end):
    for pos in range(pos_start, pos_end+1):
        np_array = reader.get_array(position=pos)
        bg = [0]*len(lambda_emiss)
        fluor_result = []
        phase_result = []
        ret_result = []
        ori_result = []
        bf_result = []
        for t in range(N_time):
            fluor_result.append(fluor_setup.deconvolve_fluor_3D(np.transpose(np_array[t,fluor_channel_idx],(0,2,3,1)).astype('float32'), bg, reg=flu_reg, autotune=False))
            S_image_recon = QLIPP_setup.Stokes_recon(np.transpose(np_array[t,QLIPP_channel_idx],(0,2,3,1)))
            S_image_tm = QLIPP_setup.Stokes_transform(S_image_recon)
            poly_order = 2
            bg_estimator = BackgroundEstimator2D()
            S_bg_extra = np.zeros_like(S_image_tm)
            S_image_tm_bg_sub = np.zeros_like(S_image_tm)
            for i in range(N_defocus):
                for p in range(5):
                    S_bg_extra[p,:,:,i] = bg_estimator.get_background(S_image_tm[p,:,:,i], order=poly_order, normalize=False)
            S_image_tm_bg_sub[0] = S_image_tm[0] / np.mean(S_bg_extra,axis=3)[0,:,:,np.newaxis]
            S_image_tm_bg_sub[1] = S_image_tm[1] - np.mean(S_bg_extra,axis=3)[1,:,:,np.newaxis]
            S_image_tm_bg_sub[2] = S_image_tm[2] - np.mean(S_bg_extra,axis=3)[2,:,:,np.newaxis]
            S_image_tm_bg_sub[3] = S_image_tm[3].copy()
            S_image_tm_bg_sub[4] = S_image_tm[4] / np.mean(S_bg_extra,axis=3)[4,:,:,np.newaxis]
            Recon_para = QLIPP_setup.Polarization_recon(S_image_tm_bg_sub)
            ret_result.append(Recon_para[0]/2/np.pi*lambda_illu*1e3)
            ori_result.append(Recon_para[1])
            bf_result.append(Recon_para[2])
            phase_result.append(QLIPP_setup.Phase_recon_3D(S_image_tm[0].astype('float32'), method='Tikhonov', reg_re=phase_reg, autotune_re=False))

        fluor_result = np.array(fluor_result)
        ret_result = np.array(ret_result)
        bf_result = np.array(bf_result)
        zarr_data_array = []
        zarr_data_array.append(np.array(phase_result))
        clims = [(-0.06,0.06)]
        zarr_data_array.append(ret_result)
        clims.append((0, np.max(ret_result)))
        zarr_data_array.append(np.array(ori_result))
        clims.append((0, np.pi))
        zarr_data_array.append(bf_result)
        clims.append((0, np.max(bf_result)))
        if len(fluor_channel_idx) == 1:
            zarr_data_array.append(fluor_result)
            clims.append((0, np.max(fluor_result)/2))
        else:
            for fl in range(len(fluor_channel_idx)):
                zarr_data_array.append(fluor_result[:,fl])
                clims.append((0, np.max(fluor_result[:,fl])/2))

        writer.init_array(pos, data_shape, chunk_size, chan_names, dtype, clims, position_name=reader.stage_positions[pos]['Label'], overwrite=False)
        zarr_data_array = np.transpose(np.array(zarr_data_array),(1,0,4,2,3))
        writer.write(zarr_data_array, p=pos)


if __name__ == '__main__':
    # Set up processes
    processes = []
    pos_per_process = int(np.ceil(N_position/N_processes))
    print('pos_per_process', pos_per_process)
    for i in range(N_processes):
        pos_start = i*pos_per_process  # if i*pos_per_process <= N_position else N_position
        pos_end = np.minimum(pos_start + pos_per_process - 1, N_position)
        print("Preparing to run positions "+str(pos_start)+'-'+str(pos_end)+' on process '+str(i))
        processes.append(mp.Process(target=process_on_single_cpu, args=(pos_start, pos_end,)))

    # Run processes
    for p in processes:
        p.start()
    for p in tqdm(processes):
        p.join()
