#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift

import waveorder as wo

from waveorder.io.writer import WaveorderWriter
from waveorder.io.reader import WaveorderReader
import zarr

# ## Initialization
# ### Experimental parameters
n_media       = 1.515                         # refractive index of the immersed media for objective (oil: 1.512, water: 1.33, air: 1)
lambda_illu   = 0.532                         # illumination wavelength (um)
mag           = 63                            # magnification of the microscope                      
NA_obj        = 1.47                          # detection NA of the objective
NA_illu       = 1.4                           # illumination NA of the condenser
N_defocus     = 96                            # number of defocus images
N_channel     = 4                             # number of Polscope channels
N_pattern     = 9                             # number of illumination patterns
z_step        = 0.25                          # z_step of the stack
z_defocus     = (np.r_[:N_defocus]-0)*z_step  # z positions of the stack
ps            = 3.45*2/mag                    # effective pixel size at the sample plane (cam pix/mag in um)
cali          = False                         # correction for S1/S2 Polscope reconstruction (does not affect phase)
bg_option     = 'global'                      # background correction method for Polscope recon (does not affect phase)
use_gpu       = True                          # option to use gpu or not (required cupy)
gpu_id        = 0                             # id of gpu to use

# ### Load sample images, background images, and calibration data
# Load data and bg

uPTI_file_name = '/gpfs/CompMicro/projects/waveorderData/data_processing/20200223_LCD_63x_147_oil_test/Kazansky_2/Anisotropic_target_small_raw.zarr'
reader = WaveorderReader(uPTI_file_name, 'zarr')
I_meas = np.transpose(reader.get_array(0),(0,1,3,4,2))
I_bg = np.squeeze(np.transpose(reader.get_array(1),(0,1,3,4,2)))

# Crop the data so that it fits in the GPU memory
I_meas = I_meas[:,:,50:250,50:250,:]
I_bg = I_bg[:,:,50:250,50:250]

# Load calibration

uPTI_file = zarr.open(uPTI_file_name, mode='a')
I_cali_mean = np.array(uPTI_file.I_cali_mean)


# source polarization, instrument matrix calibration
E_in, A_matrix, I_cali_mean = wo.instrument_matrix_and_source_calibration(I_cali_mean, handedness = 'RCP')
plt.show()

# ### Initiate the reconstruction
# setup illumination patterns
_,_,Ns,Ms,_ = I_meas.shape
xx, yy, fxx, fyy = wo.gen_coordinate((Ns, Ms), ps)


rotation_angle=[180-22.5, 225-22.5, 270-22.5, 315-22.5, 0-22.5, 45-22.5, 90-22.5, 135-22.5]
sector_angle = 45

Source_BF = wo.gen_Pupil(fxx, fyy, NA_obj/n_media/2, lambda_illu/n_media)
Source = wo.gen_sector_Pupil(fxx, fyy, NA_obj/n_media, lambda_illu/n_media, sector_angle, rotation_angle)
Source.append(Source_BF)
Source = np.array(Source)
    
# setup polarization state of the illumination
Source_PolState = np.zeros((len(Source),2), complex)

for i in range(len(Source)):
    Source_PolState[i,0] = E_in[0]
    Source_PolState[i,1] = E_in[1]
    
    
wo.plot_multicolumn(fftshift(Source,axes=(1,2)), origin='lower', num_col=5)
plt.show()

# Initiate reconstruction with experimental parameters
setup = wo.waveorder_microscopy((Ns,Ms), lambda_illu, ps, NA_obj, NA_illu, z_defocus,
                                n_media=n_media, cali=cali, bg_option=bg_option,
                                A_matrix = A_matrix,
                                phase_deconv='3D', inc_recon='3D',
                                illu_mode='Arbitrary', Source = Source,
                                Source_PolState=Source_PolState,
                                use_gpu=use_gpu, gpu_id=gpu_id)

# ## Compute Stokes volumes and visualize intensity & stokes volumes.
# convert intensity to Stokes parameters
S_image_recon = setup.Stokes_recon(I_meas[:,:,:,:,::-1])
S_bg_recon = setup.Stokes_recon(I_bg[:,:,:,:])

# background correction to all the Stokes parameter
S_image_tm = np.zeros_like(S_image_recon)
S_image_tm[0] = S_image_recon[0]/S_bg_recon[0,:,:,:,np.newaxis]-1
S_image_tm[1] = S_image_recon[1]/S_bg_recon[0,:,:,:,np.newaxis] - S_bg_recon[1,:,:,:,np.newaxis]*S_image_recon[0]/S_bg_recon[0,:,:,:,np.newaxis]**2
S_image_tm[2] = S_image_recon[2]/S_bg_recon[0,:,:,:,np.newaxis] - S_bg_recon[2,:,:,:,np.newaxis]*S_image_recon[0]/S_bg_recon[0,:,:,:,np.newaxis]**2

# ## 3D uPTI reconstruction
# ### 3D volumes of the components of scattering potential tensor
# regularization on each component of the scattering potential tensor
# in the order of [0r, 0i, 1c, 1s, 2c, 2s, 3]
# It is good to set the regularization such that (1c, 1s), (2c, 2s) have the same regularization

reg_inc = np.array([2.5, 5, 1, 1, 3, 3, 3])*1

# regulairzation for estimating principal retardance
reg_ret_pr = 1e-2

# reconstruct components of scattering potential tensor
f_tensor = setup.scattering_potential_tensor_recon_3D_vec(S_image_tm, reg_inc=reg_inc, cupy_det=True)

# browse the z-slice of components of scattering potential tensor
wo.plot_multicolumn(f_tensor[...,44], num_col=4, origin='lower', size=5,
                    set_title=True, titles=[r'$f_{0r}$', r'$f_{0i}$', r'$f_{1c}$',r'$f_{1s}$',\
                                            r'$f_{2c}$', r'$f_{2s}$', r'$f_{3}$'])
plt.show()

# reconstruct 3D anisotropy (principal retardance, 3D orientation, optic sign probability)
# material type: 
# "positive" -> only solution of positively uniaxial material
# "negative" -> only solution of negatively uniaxial meterial
# "unknown" -> both solutions of positively and negatively uniaxial material + optic sign estimation

retardance_pr, azimuth, theta, mat_map = setup.scattering_potential_tensor_to_3D_orientation(f_tensor, S_image_tm,
                                                                                             material_type='unknown', reg_ret_pr = reg_ret_pr, itr=20, fast_gpu_mode=True)
plt.show()

p_mat_map = wo.optic_sign_probability(mat_map, mat_map_thres=0.2)
phase = wo.phase_inc_correction(f_tensor[0], retardance_pr[1], theta[1])
phase_PT, absorption_PT, retardance_pr_PT = [wo.unit_conversion_from_scattering_potential_to_permittivity(SP_array, lambda_illu, n_media=n_media, imaging_mode = '3D') 
                                             for SP_array in [phase, f_tensor[1].copy(), retardance_pr]]
retardance_pr_PT = np.array([((-1)**i)*wo.wavelet_softThreshold(((-1)**i)*retardance_pr_PT[i], 'db8', 0.00303, level=1) for i in range(2)])

# ## Visualize reconstructed physical properties of the anisotropic glass target
# ### Reconstructed phase, absorption, principal retardance, azimuth, and inclination assuming (+) and (-) optic sign
# browse the reconstructed physical properties
wo.plot_multicolumn(np.stack([phase_PT[...,44], retardance_pr_PT[0,:,:,44], azimuth[0,:,:,4], theta[0,:,:,44], \
                                             absorption_PT[...,44], retardance_pr_PT[1,:,:,44], azimuth[1,:,:,44], theta[1,:,:,44]]),
                    num_col=4, origin='lower', set_title=True, size=5, \
                    titles=[r'phase',r'principal retardance (+)', r'$\omega$ (+)', r'$\theta$ (+)',\
                            r'absorption',r'principal retardance (-)', r'$\omega$ (-)', r'$\theta$ (-)'])
plt.show()

# save results to zarr array
writer = WaveorderWriter('.', hcs=False, hcs_meta=None, verbose=True)
writer.create_zarr_root('Anisotropic_target_small_processed.zarr')
chan_names = ['f_tensor0r', 'f_tensor0i', 'f_tensor1c','f_tensor1s','f_tensor2c','f_tensor2s', 'f_tensor3', 'mat_map0', 'mat_map1']
uPTI_array = np.transpose(np.concatenate((f_tensor, mat_map),axis=0)[np.newaxis,...],(0,1,4,2,3)) # dimension (T, C, Z, Y, X)
data_shape = uPTI_array.shape
chunk_size = (1,1,1)+uPTI_array.shape[3:]
writer.init_array(0, data_shape, chunk_size, chan_names, position_name='f_tensor', overwrite=True)
writer.write(uPTI_array, p=0)

chan_names_phys = ['Phase3D', 'Retardance3D', 'Orientation', 'Inclination', 'Optic_sign']
phys_data_array = np.transpose(np.array([phase_PT, np.abs(retardance_pr_PT[1]), azimuth[1], theta[1], p_mat_map]),(0,3,1,2))[np.newaxis,...]
data_shape_phys = phys_data_array.shape
dtype = 'float32'
writer.init_array(1, data_shape_phys, chunk_size, chan_names_phys, dtype, position_name='Stitched_physical', overwrite=True)
writer.write(phys_data_array, p=1)


# # Load the processed results

# uPTI_file_name = 'Anisotropic_target_small_processed.zarr'
# reader = WaveorderReader(uPTI_file_name, 'zarr')

# uPTI_array = np.transpose(np.squeeze(np.array(reader.get_zarr(0))),(0,2,3,1))
# f_tensor = uPTI_array_stitched[:7]
# mat_map = uPTI_array_stitched[7:]

# # compute the physical properties from the scattering potential tensor

# retardance_pr_p, azimuth_p, theta_p = wo.scattering_potential_tensor_to_3D_orientation_PN(f_tensor, material_type='positive', reg_ret_pr = reg_ret_pr)
# retardance_pr_n, azimuth_n, theta_n = wo.scattering_potential_tensor_to_3D_orientation_PN(f_tensor, material_type='negative', reg_ret_pr = reg_ret_pr)
# retardance_pr = np.array([retardance_pr_p,retardance_pr_n])
# azimuth = np.array([azimuth_p,azimuth_n])
# theta = np.array([theta_p, theta_n])

# p_mat_map = wo.optic_sign_probability(mat_map, mat_map_thres=0.09)
# phase = wo.phase_inc_correction(f_tensor[0], retardance_pr[1], theta[1])
# phase_PT, absorption_PT, retardance_pr_PT = [wo.unit_conversion_from_scattering_potential_to_permittivity(SP_array, lambda_illu, n_media=n_media, imaging_mode = '3D') 
#                                              for SP_array in [phase, f_tensor[1].copy(), retardance_pr]]
# retardance_pr_PT = np.array([((-1)**i)*wo.wavelet_softThreshold(((-1)**i)*retardance_pr_PT[i], 'db8', 0.00303, level=1) for i in range(2)])


# ### Phase, principal retardance, azimuth, inclination, and optic signtruction
z_layer = 44
y_layer = 109
phase_min = -0.02
phase_max = 0.02
abs_min = -0.005
abs_max = 0.005
ret_min = 0
ret_max = 0.015
p_min   = 0.3
p_max   = 0.7


fig,ax = plt.subplots(6,2,figsize=(10,30))
sub_ax = ax[0,0].imshow(absorption_PT[:,:,z_layer], cmap='gray', origin='lower', vmin=abs_min, vmax=abs_max)
ax[0,0].set_title('absorption (xy)')
plt.colorbar(sub_ax, ax=ax[0,0])

sub_ax = ax[0,1].imshow(np.transpose(absorption_PT[y_layer,:,:]), cmap='gray', origin='lower',vmin=abs_min, vmax=abs_max,aspect=z_step/ps)
ax[0,1].set_title('absorption (xz)')
plt.colorbar(sub_ax, ax=ax[0,1])

sub_ax = ax[1,0].imshow(phase_PT[:,:,z_layer], cmap='gray', origin='lower', vmin=phase_min, vmax=phase_max)
ax[1,0].set_title('phase (xy)')
plt.colorbar(sub_ax, ax=ax[1,0])

sub_ax = ax[1,1].imshow(np.transpose(phase_PT[y_layer,:,:]), cmap='gray', origin='lower',vmin=phase_min, vmax=phase_max,aspect=z_step/ps)
ax[1,1].set_title('absorption (xz)')
plt.colorbar(sub_ax, ax=ax[1,1])

sub_ax = ax[2,0].imshow(np.abs(retardance_pr_PT[0,:,:,z_layer]), cmap='gray', origin='lower',vmin=ret_min, vmax=ret_max)
ax[2,0].set_title('principal retardance (+) (xy)')
plt.colorbar(sub_ax, ax=ax[2,0])

sub_ax = ax[2,1].imshow(np.transpose(np.abs(retardance_pr_PT[0,y_layer,:,:])), cmap='gray', origin='lower',vmin=ret_min, vmax=ret_max,aspect=z_step/ps)
ax[2,1].set_title('principal retardance (+) (xz)')
plt.colorbar(sub_ax, ax=ax[2,1])

sub_ax = ax[3,0].imshow(np.abs(p_mat_map[:,:,z_layer]), cmap='gray', origin='lower',vmin=p_min, vmax=p_max)
ax[3,0].set_title('optic sign probability (xy)')
plt.colorbar(sub_ax, ax=ax[3,0])

sub_ax = ax[3,1].imshow(np.transpose(np.abs(p_mat_map[y_layer,:,:])), cmap='gray', origin='lower',vmin=p_min, vmax=p_max, aspect=z_step/ps)
ax[3,1].set_title('optic sign probability (xz)')
plt.colorbar(sub_ax, ax=ax[3,1])

sub_ax = ax[4,0].imshow(azimuth[0,:,:,z_layer], cmap='gray', origin='lower',vmin=0, vmax=np.pi)
ax[4,0].set_title('in-plane orientation (+) (xy)')

sub_ax = ax[4,1].imshow(np.transpose(azimuth[0,y_layer,:,:]), cmap='gray', origin='lower', vmin=0, vmax=np.pi, aspect=z_step/ps)
ax[4,1].set_title('in-plane orientation (+) (xz)')

sub_ax = ax[5,0].imshow(theta[0,:,:,z_layer], cmap='gray', origin='lower',vmin=0, vmax=np.pi)
ax[5,0].set_title('inclination (+) (xy)')

sub_ax = ax[5,1].imshow(np.transpose(theta[0,y_layer,:,:]), cmap='gray', origin='lower', vmin=0, vmax=np.pi, aspect=z_step/ps)
ax[5,1].set_title('inclination (+) (xz)')
plt.show()

# ### Render 3D orientation with 3D colorsphere (azimuth and inclination)
# create color-coded orientation images

ret_min_color = 0
ret_max_color = 0.012

orientation_3D_image = np.transpose(np.array([azimuth[1]/2/np.pi, theta[1], (np.clip(np.abs(retardance_pr_PT[1]),ret_min_color,ret_max_color)-ret_min_color)/(ret_max_color-ret_min_color)]),(3,1,2,0))
orientation_3D_image_RGB = wo.orientation_3D_to_rgb(orientation_3D_image, interp_belt = 20/180*np.pi, sat_factor = 1)


plt.figure(figsize=(10,10))
plt.imshow(orientation_3D_image_RGB[z_layer], origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(orientation_3D_image_RGB[:,y_layer], origin='lower',aspect=z_step/ps)

# plot the top view of 3D orientation colorsphere
plt.figure(figsize=(3,3))
wo.orientation_3D_colorwheel(wheelsize=256, circ_size=50, interp_belt=20/180*np.pi, sat_factor=1)
plt.show()


# ### Render 3D orientation with 2 channels (in-plane orientation and out-of-plane tilt)
# in-plane orientation
from matplotlib.colors import hsv_to_rgb

I_hsv = np.transpose(np.array([(azimuth[1])%np.pi/np.pi,
                               np.ones_like(retardance_pr_PT[1]),
                               (np.clip(np.abs(retardance_pr_PT[1]),ret_min_color,ret_max_color)-ret_min_color)/(ret_max_color-ret_min_color)]), (3,1,2,0))
in_plane_orientation = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(10,10))
plt.imshow(in_plane_orientation[z_layer], origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(in_plane_orientation[:,y_layer], origin='lower',aspect=z_step/ps)
plt.figure(figsize=(3,3))
wo.orientation_2D_colorwheel()
plt.show()

# out-of-plane tilt
threshold_inc = np.pi/90
I_hsv = np.transpose(np.array([(-np.maximum(0,np.abs(theta[1]-np.pi/2)-threshold_inc)+np.pi/2+threshold_inc)/np.pi,
                               np.ones_like(retardance_pr_PT[1]),
                               (np.clip(np.abs(retardance_pr_PT[1]),ret_min_color,ret_max_color)-ret_min_color)/(ret_max_color-ret_min_color)]), (3,1,2,0))
out_of_plane_tilt = hsv_to_rgb(I_hsv.copy())

plt.figure(figsize=(10,10))
plt.imshow(out_of_plane_tilt[z_layer], origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(out_of_plane_tilt[:,y_layer], origin='lower',aspect=z_step/ps)
plt.show()

# ### Angular histogram of computed 3D orientation
spacing = 4
z_layer =  44


fig,ax = plt.subplots(1,1,figsize=(15,15))
wo.plot3DVectorField(np.abs(retardance_pr_PT[1,:,:,z_layer]), azimuth[1,:,:,z_layer], theta[1,:,:,z_layer], 
                     anisotropy=40*np.abs(retardance_pr_PT[1,:,:,z_layer]), cmapImage='gray', clim=[ret_min, ret_max], aspect=1, 
                     spacing=spacing, window=spacing, linelength=spacing*1.8, linewidth=1.3, cmapAzimuth='hsv', alpha=0.4)
plt.show()

ret_mask = np.abs(retardance_pr_PT[1]).copy()
ret_mask[ret_mask<0.0075]=0
ret_mask[ret_mask>0.0075]=1

plt.figure(figsize=(10,10))
plt.imshow(ret_mask[:,:,z_layer], cmap='gray', origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(ret_mask[y_layer,:,:]), cmap='gray', origin='lower', aspect=z_step/ps)
plt.show()

# Angular histogram of 3D orientation
wo.orientation_3D_hist(azimuth[1].flatten(),
                       theta[1].flatten(),
                       ret_mask.flatten(),
                       bins=72, num_col=1, size=10, contour_level = 100, hist_cmap='gray', top_hemi=True)
plt.show()
