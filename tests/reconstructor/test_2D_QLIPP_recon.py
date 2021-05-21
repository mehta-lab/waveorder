import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import pickle
import waveorder as wo



def test_2D_QLIPP_recon():
    
    """
    Test the whole pipeline of 2D QLIPP simulation and reconstruction using waveorder
    
    """
    
    # Simulation parameters

    N           = 256                    # number of pixel in y dimension
    M           = 256                    # number of pixel in x dimension
    mag         = 40                     # magnification
    ps          = 6.5/mag                # effective pixel size
    lambda_illu = 0.532                  # wavelength
    n_media     = 1                      # refractive index in the media
    NA_obj      = 0.55                   # objective NA
    NA_illu     = 0.4                    # illumination NA (condenser)
    NA_illu_in  = 0.4                    # illumination NA (phase contrast inner ring)
    z_defocus   = (np.r_[:5]-2)*1.757    # a set of defocus plane
    chi         = 0.03*2*np.pi           # swing of Polscope analyzer

    star, theta, _ = wo.genStarTarget(N,M)

    # Assign uniform phase, uniform retardance, and radial slow axes to the star pattern

    phase_value = 1 # average phase in radians (optical path length)

    phi_s = star*(phase_value + 0.15) # slower OPL across target
    phi_f = star*(phase_value - 0.15) # faster OPL across target

    mu_s = np.zeros((N,M)) # absorption
    mu_f = mu_s.copy()

    t_eigen = np.zeros((2, N, M), complex) # complex specimen transmission

    t_eigen[0] = np.exp(-mu_s + 1j*phi_s) 
    t_eigen[1] = np.exp(-mu_f + 1j*phi_f) 

    sa = theta%np.pi #slow axes.

    # Subsample source pattern for speed

    xx, yy, fxx, fyy = wo.gen_coordinate((N, M), ps)
    Source_cont = wo.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)


    Source_discrete = wo.Source_subsample(Source_cont, lambda_illu*fxx, lambda_illu*fyy, subsampled_NA = 0.1)
    
    # initiate simulator
    simulator = wo.waveorder_microscopy_simulator((N,M), lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi, n_media=n_media,\
                                                  illu_mode='Arbitrary', Source=Source_discrete)

    I_meas, _ = simulator.simulate_waveorder_measurements(t_eigen, sa, multiprocess=False)


    # initiate reconstructor
    setup = wo.waveorder_microscopy((N,M), lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi, n_media=n_media, 
                                    phase_deconv='2D', bire_in_plane_deconv='2D', illu_mode='BF')


    S_image_recon = setup.Stokes_recon(I_meas)
    S_image_tm = setup.Stokes_transform(S_image_recon)
    Recon_para = setup.Polarization_recon(S_image_tm) # Without accounting for diffraction


    # Tikhonov regularizer for phase
    reg_u = 1e-5
    reg_p = 1e-5
    S0_stack = S_image_recon[0].copy()

    mu_sample, phi_sample = setup.Phase_recon(S0_stack, method='Tikhonov', reg_u = reg_u, reg_p = reg_p)


    # TV regularizer for phase
    lambda_u = 3e-3
    lambda_p = 1e-3
    S0_stack = S_image_recon[0].copy()

    mu_sample_TV, phi_sample_TV = setup.Phase_recon(S0_stack, method='TV', lambda_u = lambda_u, lambda_p = lambda_p, itr = 10, rho=1)


    # Diffraction aware reconstruction assuming slowly varying transmission
    S1_stack = S_image_recon[1].copy()/S_image_recon[0].mean()
    S2_stack = S_image_recon[2].copy()/S_image_recon[0].mean()

    # Tikhonov
    retardance, azimuth = setup.Birefringence_recon_2D(S1_stack, S2_stack, method='Tikhonov', reg_br = 1e-3)

    # TV
    retardance_TV, azimuth_TV = setup.Birefringence_recon_2D(S1_stack, S2_stack, method='TV', reg_br = 1e-1,\
                                                          rho = 1e-5, lambda_br=1e-3, itr = 20, verbose=True)


    # Compute ground truth images

    phase_gt = (phi_s + phi_f)/2
    phase_gt = np.real(ifft2(np.mean(fft2(phase_gt)[:,:,np.newaxis]*np.abs(setup.Hp)**2 / (np.abs(setup.Hp)**2+reg_p),axis=2)))
    phase_gt -= np.mean(phase_gt)

    S1_gt = (phi_s - phi_f)*np.sin(2*sa)
    S2_gt = (phi_s - phi_f)*np.cos(2*sa)


    # test the accuracy of reconstruction using the relative error

    # phase_Tikhonov (sensitive to regularization tuning, might cause error if the regularization is not tuned properly)
    assert(np.sum(np.abs(phase_gt-(phi_sample-np.mean(phi_sample)))**2)/np.sum(np.abs(phase_gt)**2) < 0.1)

    # phase_TV
    assert(np.sum(np.abs(phase_gt-(phi_sample_TV-np.mean(phi_sample_TV)))**2)/np.sum(np.abs(phase_gt)**2) < 0.1)

    # retardance + orientation (no deconvolution)
    assert(np.sum(np.abs(S1_gt-Recon_para[0,:,:,2]*np.sin(2*Recon_para[1,:,:,2]))**2)/np.sum(np.abs(S1_gt)**2) < 0.1)
    assert(np.sum(np.abs(S2_gt-Recon_para[0,:,:,2]*np.cos(2*Recon_para[1,:,:,2]))**2)/np.sum(np.abs(S2_gt)**2) < 0.1)

    # retardance + orientation (Tikhonov)
    assert(np.sum(np.abs(S1_gt-retardance*np.sin(2*azimuth))**2)/np.sum(np.abs(S1_gt)**2) < 0.1)
    assert(np.sum(np.abs(S2_gt-retardance*np.cos(2*azimuth))**2)/np.sum(np.abs(S2_gt)**2) < 0.1)

    # retardance + orientation (TV)
    assert(np.sum(np.abs(S1_gt-retardance_TV*np.sin(2*azimuth_TV))**2)/np.sum(np.abs(S1_gt)**2) < 0.1)
    assert(np.sum(np.abs(S2_gt-retardance_TV*np.cos(2*azimuth_TV))**2)/np.sum(np.abs(S2_gt)**2) < 0.1)