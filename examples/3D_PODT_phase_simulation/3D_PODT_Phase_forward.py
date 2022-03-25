#!/usr/bin/env python
# coding: utf-8
#################################################################################################################
# 3D Partially coherent ODT forward simulation                                                                  #
# This forward simulation is based on the SEAGLE paper (https://ieeexplore.ieee.org/abstract/document/8074742)  #
# ```H.-Y. Liu, D. Liu, H. Mansour, P. T. Boufounos, L. Waller, and U. S. Kamilov, "SEAGLE: Sparsity-Driven     #
# Image Reconstruction Under Multiple Scattering," IEEE Trans. Computational Imaging vol.4, pp.73-86 (2018).``` #
# and the 3D PODT paper (https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-14-15699&id=368361):           #
# ```J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative 3D tomographic imaging for partially    #
# coherent light microscopy," Opt. Express 25, 15699-15712 (2017).```                                           #
#################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
import waveorder as wo

# ### Experiment parameters
N           = 256                 # number of pixel in y dimension
M           = 256                 # number of pixel in x dimension
L           = 100                 # number of layers in z dimension
n_media     = 1.46                # refractive index in the media
mag         = 63                  # magnification
ps          = 6.5/mag             # effective pixel size
psz         = 0.25                # axial pixel size
lambda_illu = 0.532               # wavelength
NA_obj      = 1.2                 # objective NA
NA_illu     = 0.9                 # illumination NA

# ### Sample creation
radius = 5
blur_size = 2*ps
sphere, _, _ = wo.gen_sphere_target((N,M,L), ps, psz, radius, blur_size)
wo.image_stack_viewer(np.transpose(sphere,(2,0,1)))

# Physical value assignment

n_sample = 1.50 

RI_map = np.zeros_like(sphere)
RI_map[sphere > 0] = sphere[sphere > 0]*(n_sample-n_media)
RI_map += n_media
t_obj = np.exp(1j*2*np.pi*psz*(RI_map-n_media))

plt.figure(figsize=(10,10))
plt.imshow(np.angle(t_obj[:,:,L//2]), cmap='gray', origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(np.angle(t_obj[N//2,:,:])), cmap='gray', origin='lower',aspect=psz/ps)
plt.show()

# ### Setup acquisition
# Subsampled Source pattern

xx, yy, fxx, fyy = wo.gen_coordinate((N, M), ps)
Source_cont = wo.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)
Source_discrete = wo.Source_subsample(Source_cont, lambda_illu*fxx, lambda_illu*fyy, subsampled_NA = 0.1)
plt.figure(figsize=(10,10))
plt.imshow(fftshift(Source_discrete),cmap='gray')
plt.show()
print(np.sum(Source_discrete))

z_defocus = (np.r_[:L]-L//2)*psz
chi = 0.1*2*np.pi
setup = wo.waveorder_microscopy((N,M), lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi,
                                n_media = n_media, phase_deconv='3D', illu_mode='Arbitrary', Source=Source_cont)

simulator = wo.waveorder_microscopy_simulator((N,M), lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi,
                                              n_media = n_media, illu_mode='Arbitrary', Source=Source_discrete)

plt.figure(figsize=(5,5))
plt.imshow(fftshift(setup.Source), cmap='gray')
plt.colorbar()
H_re_vis = fftshift(setup.H_re)
wo.plot_multicolumn([np.real(H_re_vis)[:,:,L//2], np.transpose(np.real(H_re_vis)[N//2,:,:]),
                     np.imag(H_re_vis)[:,:,L//2], np.transpose(np.imag(H_re_vis)[N//2,:,:])],
                    num_col=2, size=8, set_title=True,
                    titles=['$xy$-slice of Re{$H_{re}$} at $u_z=0$', '$xz$-slice of Re{$H_{re}$} at $u_y=0$',
                            '$xy$-slice of Im{$H_{re}$} at $u_z=0$', '$xz$-slice of Im{$H_{re}$} at $u_y=0$'], colormap='jet')
plt.show()

H_im_vis = fftshift(setup.H_im)
wo.plot_multicolumn([np.real(H_im_vis)[:,:,L//2], np.transpose(np.real(H_im_vis)[N//2,:,:]),
                     np.imag(H_im_vis)[:,:,L//2], np.transpose(np.imag(H_im_vis)[N//2,:,:])],
                    num_col=2, size=8, set_title=True,
                    titles=['$xy$-slice of Re{$H_{im}$} at $u_z=0$', '$xz$-slice of Re{$H_{im}$} at $u_y=0$',
                            '$xy$-slice of Im{$H_{im}$} at $u_z=0$', '$xz$-slice of Im{$H_{im}$} at $u_y=0$'], colormap='jet')
plt.show()


I_meas = simulator.simulate_3D_scalar_measurements(t_obj)
plt.figure(figsize=(10,10))
plt.imshow(I_meas[:,:,L//2], cmap='gray', origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(I_meas[N//2,:,:]), cmap='gray', origin='lower',aspect=psz/ps)
plt.show()


# Save simulations
output_file = '/data_sm/home/lihao/project/Polscope/Simulation/3D_Pol_Phase/uPTI_repo_demo/3D_PODT_simulation'
np.savez(output_file, I_meas=I_meas, lambda_illu=lambda_illu,
         n_media=n_media, NA_obj=NA_obj, NA_illu=NA_illu, ps=ps, psz=psz, Source_cont=Source_cont)




