# script to take multiple ometiff input folders, perform deconvolutions for each channel, and write a single 
# omezarr output. The current script performs phase reconstruct from 'BF' channel and deconvolution of two 
# fluorescence channels 'DAPI' and 'Y5'. 

# %% import libraries
import os 
import numpy as np
import waveorder as wo
from waveorder.io import WaveorderReader, WaveorderWriter

# %% define the inputs and initialize variables
input_folder = '/hpc/projects/compmicro/rawdata/hummingbird/Soorya/2022_11_01_VeroMemNuclStain/'
output_folder = './2022_11_01_VeroMemNuclStain/'
data_folders = os.listdir(input_folder)

# phase recon parameter
ps_f = 3.45 / 20  # effective pixel size at the sample plane (cam pix/mag in um)
n_media = 1  # refractive index of the immersed media for objective (oil: 1.512, water: 1.33, air: 1)
NA_obj = 0.55  # detection NA of the objective
lambda_illu = 0.532  # illumination wavelength (um)
NA_illu = 0.4  # illumination NA of the condenser

# deconvolution parameters
pad_z = 5
use_gpu = False
gpu_id = 0

# fluorescence deconvolution parameters
t=0         # current data has only one time point. Uncomment time loop in section below if multiple time point data
bg=[50]     # fluorescence background specific to Hummingbird microscope
FL1_name = 'DAPI'  # first fluorecence channel name (check from MM read params)
lambda_emiss_FL1 = [0.467]  # emission wavelength of DAPI fluorescence channel (list, in um)
FL2_name = 'Y5'  # second fluorecence channel name (check from MM read params)
lambda_emiss_FL2 = [0.666]  # Y5 -cellmask emission wavelength

# Get total number of positions, data shape, channel names

# %% Make a single zarr 
writer = WaveorderWriter(output_folder, hcs=False, hcs_meta=None, verbose=True)
writer.create_zarr_root('output.zarr')

# %% Deconvolve different channels from multiple ome.tiffs & write into the zarr store
i = 0
for data_folder in data_folders:
    reader = WaveorderReader(input_folder + data_folder)
    data_shape = reader.shape # assumed to be the same for all positions
    T, C, Z, Y, X = data_shape    
    chan_names = reader.channel_names # ""
    dtype = 'float32' #reader.dtype # ""
    psz = reader.z_step_size  # z-step size of the defocused intensity stack (in um)
    N_defocus = reader.slices
    z_defocus = -(np.r_[:N_defocus] - 0) * psz  # z positions of the stack

    for pos in range(reader.get_num_positions()):
        writer.init_array(i, data_shape, chunk_size=(1,1,1,Y,X), chan_names=chan_names, dtype=dtype, clims=len(chan_names)*[(0,1)])

        #import pdb; pdb.set_trace()   # activate for debugging
        raw_data = reader.get_array(pos)

        # phase reconstruction from BF channel

        BF = raw_data[0,reader.channel_names.index('BF'),...]

        phase_setup = wo.waveorder_microscopy((Y, X), lambda_illu, ps_f, NA_obj, NA_illu, z_defocus,
		                                  n_media=n_media, cali=False, bg_option='global', chi=np.pi / 2,
		                                  phase_deconv='3D', pad_z=pad_z,
		                                  use_gpu=use_gpu, gpu_id=gpu_id)

        # #for t in range(N_time):   # use for multiple time point data
        phase_result = phase_setup.Phase_recon_3D(
            np.transpose(BF, (1, 2, 0)).astype('float32'),
            method='Tikhonov', reg_re=1e-02, autotune_re=False)

        writer.write(phase_result.transpose(2, 0, 1), p=i, t=t, c=chan_names.index('BF'))

        
         # deconvolution of first fluorescence channel, 'DAPI'

        FL1 = raw_data[0,reader.channel_names.index(FL1_name),...]

        fluor_setup = wo.fluorescence_microscopy((Y, X, N_defocus), lambda_emiss_FL1, ps_f, psz, NA_obj,
                                                n_media=n_media, deconv_mode='3D-WF', pad_z=pad_z, use_gpu=use_gpu,
                                                gpu_id=gpu_id)

        #for t in range(N_time):
        fluor_result = fluor_setup.deconvolve_fluor_3D(
            np.transpose(FL1, (1, 2, 0)).astype('float32'), bg, reg=[1e-3],autotune=False)


        writer.write(fluor_result.transpose(2, 0, 1), p=i, t=t, c=chan_names.index(FL1_name))


        # deconvolve second fluorescence channel, here Y5

        FL2 = raw_data[0,reader.channel_names.index(FL2_name),...]

        fluor_setup = wo.fluorescence_microscopy((Y, X, N_defocus), lambda_emiss_FL2, ps_f, psz, NA_obj,
                                                n_media=n_media, deconv_mode='3D-WF', pad_z=pad_z, use_gpu=use_gpu,
                                                gpu_id=gpu_id)

        #for t in range(N_time):
        fluor_result = fluor_setup.deconvolve_fluor_3D(
            np.transpose(FL2, (1, 2, 0)).astype('float32'), bg, reg=[1e-3],autotune=False)


        writer.write(fluor_result.transpose(2, 0, 1), p=i, t=t, c=chan_names.index(FL2_name))

        
        #writer.write(deconvolved_data, p=i)
        i += 1

# %%
