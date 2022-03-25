#!/usr/bin/env python
# coding: utf-8

#################################################################################################################
# 3D Partially coherent ODT processing                                                                  #
# This forward simulation is based on the SEAGLE paper (https://ieeexplore.ieee.org/abstract/document/8074742)  #
# ```H.-Y. Liu, D. Liu, H. Mansour, P. T. Boufounos, L. Waller, and U. S. Kamilov, "SEAGLE: Sparsity-Driven     #
# Image Reconstruction Under Multiple Scattering," IEEE Trans. Computational Imaging vol.4, pp.73-86 (2018).``` #
# and the 3D PODT paper (https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-14-15699&id=368361):           #
# ```J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative 3D tomographic imaging for partially    #
# coherent light microscopy," Opt. Express 25, 15699-15712 (2017).```                                           #
#################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift

import waveorder as wo

# ### Load data
# Load simulations
file_name = '/data_sm/home/lihao/project/Polscope/Simulation/3D_Pol_Phase/uPTI_repo_demo/3D_PODT_simulation.npz'
array_loaded = np.load(file_name)
list_of_array_names = sorted(array_loaded)
for array_name in list_of_array_names:
    globals()[array_name] = array_loaded[array_name]

print(list_of_array_names)
N, M, L = I_meas.shape


# ### Refractive index reconstruction
z_defocus = (np.r_[:L]-L//2)*psz
chi = 0.1*2*np.pi
setup = wo.waveorder_microscopy((N,M), lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi,
                                n_media = n_media, phase_deconv='3D', pad_z=10)

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

plt.figure(figsize=(10,10))
plt.imshow(I_meas[:,:,L//2], cmap='gray', origin='lower')
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(I_meas[N//2,:,:]), cmap='gray', origin='lower',aspect=psz/ps)
plt.show()

f_real = setup.Phase_recon_3D(I_meas, absorption_ratio=0.0, method='Tikhonov', reg_re = 1e-4)

# visualization of the phase images
plt.figure(figsize=(10,10))
plt.imshow(f_real[:,:,L//2], cmap='gray', origin='lower')
plt.show()
plt.figure(figsize=(10,10))
plt.imshow(np.transpose(f_real[N//2,:,:]), cmap='gray', origin='lower',aspect=psz/ps)
plt.show()
plt.plot(f_real[N//2,:,L//2])
plt.show()



