import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import gc
import os
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import uniform_filter
from concurrent.futures import ProcessPoolExecutor
from .util import *
from .optics import *
from .background_estimator import *

def intensity_mapping(img_stack):
    img_stack_out = np.zeros_like(img_stack)
    img_stack_out[0] = img_stack[0].copy()
    img_stack_out[1] = img_stack[4].copy()
    img_stack_out[2] = img_stack[3].copy()
    img_stack_out[3] = img_stack[1].copy()
    img_stack_out[4] = img_stack[2].copy()
    
    return img_stack_out


def Jones_PC_forward_model(t_eigen, sa_orientation, fxx, fyy, xx, yy, N_defocus, N_channel, analyzer_para, Pupil_obj, Hz_det, time_re):
    
    plane_wave = np.exp(1j*2*np.pi*(fyy * yy +\
                                    fxx * xx))
    
    N, M = xx.shape
    E_field = []
    E_field.append(plane_wave)
    E_field.append(1j*plane_wave) # RHC illumination
    E_field = np.array(E_field)

    E_sample = Jones_sample(E_field, t_eigen, sa_orientation)

    Stokes_ang = np.zeros((4, N, M, N_defocus))
    I_meas_ang = np.zeros((N_channel, N, M, N_defocus))


    for m in range(N_defocus):
        Pupil_eff = Pupil_obj * Hz_det[:,:,m]
        E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
        Stokes_ang[:,:,:,m] = Jones_to_Stokes(E_field_out)

        for n in range(N_channel):
            I_meas_ang[n,:,:,m] = np.abs(analyzer_output(E_field_out, analyzer_para[n,0], analyzer_para[n,1]))**2
            
            
    print('Processed %d, elapsed time: %.2f'%(os.getpid(), time.time() - time_re))

    return (Stokes_ang, I_meas_ang)


class waveorder_microscopy:
    
    def __init__(self, img_dim, lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi,\
                 n_media=1, cali=False, bg_option='global', 
                 phase_deconv='2D', ph_deconv_layer = 5,
                 illu_mode='BF', NA_illu_in=None, Source=None, 
                 use_gpu=False, gpu_id=0):
        
        '''
        
        initialize the system parameters for phase and orders microscopy            
        
        '''
        
        t0 = time.time()
        
        # GPU/CPU
        
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        if self.use_gpu:
            globals()['cp'] = __import__("cupy")
            cp.cuda.Device(self.gpu_id).use()
            
        
        # Basic parameter 
        self.N, self.M   = img_dim
        self.n_media     = n_media
        self.lambda_illu = lambda_illu/n_media
        self.ps          = ps
        self.z_defocus   = z_defocus.copy()
        if len(z_defocus) >= 2:
            self.psz     = np.abs(z_defocus[0] - z_defocus[1])
        self.NA_obj      = NA_obj/n_media
        self.NA_illu     = NA_illu/n_media
        self.N_defocus   = len(z_defocus)
        self.chi         = chi
        self.cali        = cali
        self.bg_option   = bg_option
        
        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        self.Pupil_obj = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.Pupil_support = self.Pupil_obj.copy()
        
        # illumination setup
        
        if illu_mode == 'BF':
            self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
            self.N_pattern = 1
        
        elif illu_mode == 'PH':
            if NA_illu_in == None:
                raise('No inner rim NA specified in the PH illumination mode')
            else:
                self.NA_illu_in  = NA_illu_in/n_media
                inner_pupil = gen_Pupil(self.fxx, self.fyy, self.NA_illu_in, self.lambda_illu)
                self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
                self.Source -= inner_pupil
                self.Pupil_obj = self.Pupil_obj*np.exp(self.Source*(np.log(0.5)+1j*np.pi/2))
                self.N_pattern = 1
        elif illu_mode == 'Arbitrary':
    
            self.Source = Source.copy()
            if Source.ndim == 2:
                self.N_pattern = 1
            else:
                self.N_pattern = len(Source)
                
        t_basic = time.time()
        print('Finished constructing basic parameters, elapsed time %.2f'%(t_basic-t0))

        # select either 2D or 3D model for deconvolution
        
        if phase_deconv == '2D':
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, self.z_defocus)
            self.gen_WOTF()
            
        elif phase_deconv == 'semi-2D':
            
            self.ph_deconv_layer = ph_deconv_layer
            z_deconv = -(np.r_[:self.ph_deconv_layer]-self.ph_deconv_layer//2)*self.psz
            
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv)
            self.G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv)
            self.gen_semi_2D_WOTF()
            
        elif phase_deconv == '3D':
            z = ifftshift((np.r_[0:self.N_defocus]-self.N_defocus//2)*self.psz)
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z)
            self.G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z)
            self.gen_3D_WOTF()
        
        self.analyzer_para = np.array([[np.pi/2, np.pi], \
                                       [np.pi/2-self.chi, np.pi], \
                                       [np.pi/2, np.pi-self.chi], \
                                       [np.pi/2+self.chi, np.pi], \
                                       [np.pi/2, np.pi+self.chi]]) # [alpha, beta]
        
        self.N_channel = len(self.analyzer_para)
        

    
    def gen_WOTF(self):

        self.Hu = np.zeros((self.N, self.M, self.N_defocus*self.N_pattern),complex)
        self.Hp = np.zeros((self.N, self.M, self.N_defocus*self.N_pattern),complex)
        
        if self.N_pattern == 1:
            for i in range(self.N_defocus):
                self.Hu[:,:,i], self.Hp[:,:,i] = WOTF_2D_compute(self.Source, self.Pupil_obj * self.Hz_det[:,:,i], \
                                                                 use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        else:
            
            for i,j in itertools.product(range(self.N_defocus), range(self.N_pattern)):
                idx = i*self.N_pattern+j
                self.Hu[:,:,idx], self.Hp[:,:,idx] = WOTF_2D_compute(self.Source[j], self.Pupil_obj * self.Hz_det[:,:,i], \
                                                                       use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                
    def gen_semi_2D_WOTF(self):
        
        self.Hu = np.zeros((self.N, self.M, self.ph_deconv_layer*self.N_pattern),complex)
        self.Hp = np.zeros((self.N, self.M, self.ph_deconv_layer*self.N_pattern),complex)
        
        if self.N_pattern == 1:
            
            for i in range(self.ph_deconv_layer):
                self.Hu[:,:,i], self.Hp[:,:,i] = WOTF_semi_2D_compute(self.Source, self.Pupil_obj, self.Hz_det[:,:,i], \
                                                                      self.G_fun_z[:,:,i]*4*np.pi*1j/self.lambda_illu, \
                                                                      use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                
        else:
            
            for i,j in itertools.product(range(self.ph_deconv_layer), range(self.N_pattern)):
                idx = i*self.N_pattern+j
                self.Hu[:,:,idx], self.Hp[:,:,idx] = WOTF_semi_2D_compute(self.Source[j], self.Pupil_obj, self.Hz_det[:,:,i], \
                                                                          self.G_fun_z[:,:,i]*4*np.pi*1j/self.lambda_illu, \
                                                                          use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
            
    def gen_3D_WOTF(self):
        
        self.H_re = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_im = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        
        for i in range(self.N_pattern):
            self.H_re[i], self.H_im[i] = WOTF_3D_compute(self.Source.astype('float32'), self.Pupil_obj.astype('complex64'), self.Hz_det.astype('complex64'), \
                                                         self.G_fun_z.astype('complex64'), self.psz,\
                                                         use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        self.H_re = np.squeeze(self.H_re)
        self.H_im = np.squeeze(self.H_im)
        
        
            
    def simulate_waveorder_measurements(self, t_eigen, sa_orientation, multiprocess=False):        
        
        Stokes_out = np.zeros((4, self.N, self.M, self.N_defocus*self.N_pattern))
        I_meas = np.zeros((self.N_channel, self.N, self.M, self.N_defocus*self.N_pattern))
        
        if multiprocess:
            
            
            t0 = time.time()
            for j in range(self.N_pattern):

                if self.N_pattern == 1:
                    [idx_y, idx_x] = np.where(self.Source ==1) 
                else:
                    [idx_y, idx_x] = np.where(self.Source[j] ==1)
                N_source = len(idx_y)
                
                
                
                t_eigen_re = itertools.repeat(t_eigen, N_source)
                sa_orientation_re = itertools.repeat(sa_orientation, N_source)
                fxx = self.fxx[idx_y, idx_x].tolist()
                fyy = self.fyy[idx_y, idx_x].tolist()
                xx = itertools.repeat(self.xx, N_source)
                yy = itertools.repeat(self.yy, N_source)
                N_defocus = itertools.repeat(self.N_defocus, N_source)
                N_channel = itertools.repeat(self.N_channel, N_source)
                analyzer_para = itertools.repeat(self.analyzer_para, N_source)
                Pupil_obj = itertools.repeat(self.Pupil_obj, N_source)
                Hz_det = itertools.repeat(self.Hz_det, N_source)
                time_re = itertools.repeat(t0, N_source)
                
                # Jones_PC_forward_model(t_eigen, sa_orientation, fxx, fyy, xx, yy, N_defocus, N_channel, analyzer_para, Pupil_obj, Hz_det)
                
                with ProcessPoolExecutor() as executor:
                    for result in executor.map(Jones_PC_forward_model, t_eigen_re, sa_orientation_re, \
                                               fxx, fyy, xx, yy, N_defocus, N_channel, analyzer_para, Pupil_obj, Hz_det, time_re):
                        Stokes_out += result[0]
                        I_meas_ang += result[1]
                
                print('Number of sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f'\
                              %(N_source, N_source, j+1, self.N_pattern, time.time()-t0))
                        
                        
            
            
        else:
            t0 = time.time()
            for j in range(self.N_pattern):

                if self.N_pattern == 1:
                    [idx_y, idx_x] = np.where(self.Source ==1) 
                else:
                    [idx_y, idx_x] = np.where(self.Source[j] ==1)
                N_source = len(idx_y)


                for i in range(N_source):
                    plane_wave = np.exp(1j*2*np.pi*(self.fyy[idx_y[i], idx_x[i]] * self.yy +\
                                                    self.fxx[idx_y[i], idx_x[i]] * self.xx))
                    E_field = []
                    E_field.append(plane_wave)
                    E_field.append(1j*plane_wave) # RHC illumination
                    E_field = np.array(E_field)

                    E_sample = Jones_sample(E_field, t_eigen, sa_orientation)

                    for m in range(self.N_defocus):
                        Pupil_eff = self.Pupil_obj * self.Hz_det[:,:,m]
                        E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
                        Stokes_out[:,:,:,m] += Jones_to_Stokes(E_field_out)

                        for n in range(self.N_channel):
                            I_meas[n,:,:,m] += np.abs(analyzer_output(E_field_out, self.analyzer_para[n,0], self.analyzer_para[n,1]))**2

                    if np.mod(i+1, 100) == 0 or i+1 == N_source:
                        print('Number of sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f'\
                              %(i+1,N_source, j+1, self.N_pattern, time.time()-t0))

            
        return I_meas, Stokes_out
    
    def simulate_waveorder_inc_measurements(self, n_e, n_o, dz, mu, orientation, inclination):        
        
        Stokes_out = np.zeros((4, self.N, self.M, self.N_defocus*self.N_pattern))
        I_meas = np.zeros((self.N_channel, self.N, self.M, self.N_defocus*self.N_pattern))
        
        sample_norm_x = np.sin(inclination)*np.cos(orientation)
        sample_norm_y = np.sin(inclination)*np.sin(orientation)
        sample_norm_z = np.cos(inclination)
        
        wave_x = self.lambda_illu*self.fxx
        wave_y = self.lambda_illu*self.fyy
        wave_z = (np.maximum(0,1 - wave_x**2 - wave_y**2))**(0.5)
        
        
        
        for j in range(self.N_pattern):
            
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source ==1) 
            else:
                [idx_y, idx_x] = np.where(self.Source[j] ==1)
            N_source = len(idx_y)


            for i in range(N_source):
                
                cos_alpha = sample_norm_x*wave_x[idx_y[i], idx_x[i]] + \
                            sample_norm_y*wave_y[idx_y[i], idx_x[i]] + \
                            sample_norm_z*wave_z[idx_y[i], idx_x[i]]
                
                n_e_alpha = 1/((1-cos_alpha**2)/n_e**2 + cos_alpha**2/n_o**2)**(0.5)
                
                t_eigen = np.zeros((2, self.N, self.M), complex)

                t_eigen[0] = np.exp(-mu + 1j*2*np.pi*dz*(n_e_alpha/self.n_media-1)/self.lambda_illu)
                t_eigen[1] = np.exp(-mu + 1j*2*np.pi*dz*(n_o/self.n_media-1)/self.lambda_illu)

                
                plane_wave = np.exp(1j*2*np.pi*(self.fyy[idx_y[i], idx_x[i]] * self.yy +\
                                                self.fxx[idx_y[i], idx_x[i]] * self.xx))
                E_field = []
                E_field.append(plane_wave)
                E_field.append(1j*plane_wave) # RHC illumination
                E_field = np.array(E_field)


                E_sample = Jones_sample(E_field, t_eigen, orientation)

                for m in range(self.N_defocus):
                    Pupil_eff = self.Pupil_obj * self.Hz_det[:,:,m]
                    E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
                    Stokes_out[:,:,:,m*self.N_pattern+j] += Jones_to_Stokes(E_field_out)

                    for n in range(self.N_channel):
                        I_meas[n,:,:,m*self.N_pattern+j] += np.abs(analyzer_output(E_field_out, self.analyzer_para[n,0], self.analyzer_para[n,1]))**2

                if np.mod(i+1, 100) == 0 or i+1 == N_source:
                    print('Number of sources considered (%d / %d) in pattern (%d / %d)'%(i+1,N_source, j+1, self.N_pattern))

            
        return I_meas, Stokes_out
    
    def simulate_3D_scalar_measurements(self, t_obj):
        
        fr = (self.fxx**2 + self.fyy**2)**(0.5)
        Pupil_prop = gen_Pupil(self.fxx, self.fyy, 1, self.lambda_illu)
        oblique_factor_prop = ((1 - self.lambda_illu**2 * fr**2) *Pupil_prop)**(1/2) / self.lambda_illu
        z_defocus = self.z_defocus-(self.N_defocus/2-1)*self.psz
        Hz_defocus = Pupil_prop[:,:,np.newaxis] * np.exp(1j*2*np.pi*z_defocus[np.newaxis,np.newaxis,:] *\
                                                         oblique_factor_prop[:,:,np.newaxis])
        Hz_step = Pupil_prop * np.exp(1j*2*np.pi*self.psz* oblique_factor_prop)


        I_meas = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus))
        
        t0 = time.time()
        for i in range(self.N_pattern):
            
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source ==1)
            else:
                [idx_y, idx_x] = np.where(self.Source[i] ==1)
            
            N_pt_source = len(idx_y)
            
            for j in range(N_pt_source):
                plane_wave = np.exp(1j*2*np.pi*(self.fyy[idx_y[j], idx_x[j]] * self.yy +\
                                                self.fxx[idx_y[j], idx_x[j]] * self.xx))

                for m in range(self.N_defocus):

                    if m == 0:
                        f_field = plane_wave

                    g_field = f_field * t_obj[:,:,m]

                    if m == self.N_defocus-1:

                        f_field_stack_f = fft2(g_field[:,:,np.newaxis],axes=(0,1))*Hz_defocus
                        I_meas[i] += np.abs(ifft2(f_field_stack_f * self.Pupil_obj[:,:,np.newaxis], axes=(0,1)))**2

                    else:
                        f_field = ifft2(fft2(g_field)*Hz_step)

                if np.mod(j+1, 100) == 0 or j+1 == N_pt_source:
                    print('Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f'\
                          %(j+1, N_pt_source, i+1, self.N_pattern, time.time()-t0))
            
        return np.squeeze(I_meas)
    
    
    def Stokes_recon(self, I_meas):
        
        
        A_forward = 0.5*np.array([[1,0,0,-1], \
                          [1, np.sin(self.chi), 0, -np.cos(self.chi)], \
                          [1, 0, np.sin(self.chi), -np.cos(self.chi)], \
                          [1, -np.sin(self.chi), 0, -np.cos(self.chi)], \
                          [1, 0, -np.sin(self.chi), -np.cos(self.chi)]])
        
        
        A_pinv = np.linalg.pinv(A_forward)
        Stokes_inv = lambda x:  np.transpose(np.squeeze(np.matmul(A_pinv, np.transpose(x,(1,2,0))[:,:,:,np.newaxis])),(2,0,1))
        
        
        dim = I_meas.ndim 
        
        if dim == 3:
            S_image_recon = Stokes_inv(I_meas)
        else:

            S_image_recon = np.zeros((4, self.N, self.M, self.N_defocus*self.N_pattern))
            
            for i in range(self.N_defocus*self.N_pattern):
                S_image_recon[:,:,:,i] = Stokes_inv(I_meas[:,:,:,i])

            
        return S_image_recon
    
    
    def Stokes_transform(self, S_image_recon):
        
        if self.use_gpu:
            S_image_recon = cp.array(S_image_recon)
            S_transformed = cp.zeros((5,)+S_image_recon.shape[1:])
        else:
            S_transformed = np.zeros((5,)+S_image_recon.shape[1:])
        
        S_transformed[0] = S_image_recon[0]
        S_transformed[1] = S_image_recon[1] / S_image_recon[3]
        S_transformed[2] = S_image_recon[2] / S_image_recon[3]
        S_transformed[3] = S_image_recon[3]
        S_transformed[4] = (S_image_recon[1]**2 + S_image_recon[2]**2 + S_image_recon[3]**2)**(1/2) / S_image_recon[0] # DoP
        
        if self.use_gpu:
            S_transformed = cp.asnumpy(S_transformed)
        
        return S_transformed
    
    
    def Polscope_bg_correction(self, S_image_tm, S_bg_tm, kernel_size=400, poly_order=2):
        
        if self.use_gpu:
            S_image_tm = cp.array(S_image_tm)
            S_bg_tm = cp.array(S_bg_tm)
        
        dim = S_image_tm.ndim
        if dim == 3:
            S_image_tm[0] /= S_bg_tm[0]
            S_image_tm[1] -= S_bg_tm[1]
            S_image_tm[2] -= S_bg_tm[2]
            S_image_tm[4] /= S_bg_tm[4]
        else:
            S_image_tm[0] /= S_bg_tm[0,:,:,np.newaxis]
            S_image_tm[1] -= S_bg_tm[1,:,:,np.newaxis]
            S_image_tm[2] -= S_bg_tm[2,:,:,np.newaxis]
            S_image_tm[4] /= S_bg_tm[4,:,:,np.newaxis]


 
        
        
        if self.bg_option == 'local':
            
            if dim == 3:
                S_image_tm[1] -= uniform_filter_2D(S_image_tm[1], size=kernel_size, use_gpu=self.use_gpu)
                S_image_tm[2] -= uniform_filter_2D(S_image_tm[2], size=kernel_size, use_gpu=self.use_gpu)
            else:
                if self.use_gpu:
                    S1_bg = uniform_filter_2D(cp.mean(S_image_tm[1],axis=-1), size=kernel_size, use_gpu=self.use_gpu)
                    S2_bg = uniform_filter_2D(cp.mean(S_image_tm[2],axis=-1), size=kernel_size, use_gpu=self.use_gpu)
                else:
                    S1_bg = uniform_filter_2D(np.mean(S_image_tm[1],axis=-1), size=kernel_size, use_gpu=self.use_gpu)
                    S2_bg = uniform_filter_2D(np.mean(S_image_tm[2],axis=-1), size=kernel_size, use_gpu=self.use_gpu)
                    
                
                for i in range(self.N_defocus):
                    S_image_tm[1,:,:,i] -= S1_bg
                    S_image_tm[2,:,:,i] -= S2_bg
                    
        elif self.bg_option == 'local_fit':
            if self.use_gpu:
                bg_estimator = BackgroundEstimator2D_GPU(gpu_id=self.gpu_id)
                if dim != 3:
                    S1_bg = bg_estimator.get_background(cp.mean(S_image_tm[1],axis=-1), order=poly_order, normalize=False)
                    S2_bg = bg_estimator.get_background(cp.mean(S_image_tm[2],axis=-1), order=poly_order, normalize=False)
            else:
                bg_estimator = BackgroundEstimator2D()
                if dim != 3:
                    S1_bg = bg_estimator.get_background(np.mean(S_image_tm[1],axis=-1), order=poly_order, normalize=False)
                    S2_bg = bg_estimator.get_background(np.mean(S_image_tm[2],axis=-1), order=poly_order, normalize=False)
                    
            if dim ==3:
                S_image_tm[1] -= bg_estimator.get_background(S_image_tm[1], order=poly_order, normalize=False)
                S_image_tm[2] -= bg_estimator.get_background(S_image_tm[2], order=poly_order, normalize=False)
            else:
                
                for i in range(self.N_defocus):
                    S_image_tm[1,:,:,i] -= S1_bg
                    S_image_tm[2,:,:,i] -= S2_bg
                
        
        if self.use_gpu:
            S_image_tm = cp.asnumpy(S_image_tm)
        
        return S_image_tm
    
    
    
    
    def Polarization_recon(self, S_image_recon):
        
        if self.use_gpu:
            S_image_recon = cp.array(S_image_recon)
            Recon_para = cp.zeros((4,)+S_image_recon.shape[1:])
        else:
            Recon_para = np.zeros((4,)+S_image_recon.shape[1:])
        
        
        if self.use_gpu:
            Recon_para[0] = cp.arctan2((S_image_recon[1]**2 + S_image_recon[2]**2)**(1/2) * \
                                   S_image_recon[3], S_image_recon[3])  # retardance
            if self.cali == True:
                Recon_para[1] = 0.5*cp.arctan2(-S_image_recon[1], -S_image_recon[2])%np.pi # slow-axis
            else:
                Recon_para[1] = 0.5*cp.arctan2(-S_image_recon[1], S_image_recon[2])%np.pi # slow-axis
        else:
            
            Recon_para[0] = np.arctan2((S_image_recon[1]**2 + S_image_recon[2]**2)**(1/2) * \
                                       S_image_recon[3], S_image_recon[3])  # retardance
            if self.cali == True:
                Recon_para[1] = 0.5*np.arctan2(-S_image_recon[1], -S_image_recon[2])%np.pi # slow-axis
            else:
                Recon_para[1] = 0.5*np.arctan2(-S_image_recon[1], S_image_recon[2])%np.pi # slow-axis
                
        
        Recon_para[2] = S_image_recon[0] # transmittance
        Recon_para[3] = S_image_recon[4] # DoP
        
        
        if self.use_gpu:
            Recon_para = cp.asnumpy(Recon_para)
        
        return Recon_para
    
    
    
    def inten_normalization(self, S0_stack):
        
        _,_, Nimg = S0_stack.shape
        
        if self.use_gpu:
            S0_norm_stack = cp.zeros_like(S0_stack)
            
            for i in range(Nimg):

                S0_norm_stack[:,:,i] = S0_stack[:,:,i]/uniform_filter_2D(S0_stack[:,:,i], size=self.N//2, use_gpu=True)
                S0_norm_stack[:,:,i] /= S0_norm_stack[:,:,i].mean()
                S0_norm_stack[:,:,i] -= 1

        else:
            S0_norm_stack = np.zeros_like(S0_stack)
        
            for i in range(Nimg):

                S0_norm_stack[:,:,i] = S0_stack[:,:,i]/uniform_filter(S0_stack[:,:,i], size=self.N//2)
                S0_norm_stack[:,:,i] /= S0_norm_stack[:,:,i].mean()
                S0_norm_stack[:,:,i] -= 1
            
        return S0_norm_stack
    
    def inten_normalization_3D(self, S0_stack):
        
        
            
        S0_stack_norm = np.zeros_like(S0_stack)
        S0_stack_norm = S0_stack / S0_stack.mean()
        S0_stack_norm -= 1
        
        return S0_stack_norm
    
    def Tikhonov_deconv_2D(self, AHA, b_vec):
        
        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]
            
            
        if self.use_gpu:
            mu_sample = cp.asnumpy(cp.real(cp.fft.ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant)))
            phi_sample = cp.asnumpy(cp.real(cp.fft.ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant)))
        else:
            mu_sample = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
            phi_sample = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))
            
        return mu_sample, phi_sample
    
    def ADMM_TV_deconv_2D(self, AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose):
        
        
        # ADMM deconvolution with anisotropic TV regularization
            
            
        if self.use_gpu:
            Dx = cp.zeros((self.N, self.M))
            Dy = cp.zeros((self.N, self.M))
            Dx[0,0] = 1; Dx[0,-1] = -1; Dx = cp.fft.fft2(Dx);
            Dy[0,0] = 1; Dy[-1,0] = -1; Dy = cp.fft.fft2(Dy);

            rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy)

            z_para = cp.zeros((4, self.N, self.M))
            u_para = cp.zeros((4, self.N, self.M))
            D_vec = cp.zeros((4, self.N, self.M))


        else:
            Dx = np.zeros((self.N, self.M))
            Dy = np.zeros((self.N, self.M))
            Dx[0,0] = 1; Dx[0,-1] = -1; Dx = fft2(Dx);
            Dy[0,0] = 1; Dy[-1,0] = -1; Dy = fft2(Dy);

            rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy)

            z_para = np.zeros((4, self.N, self.M))
            u_para = np.zeros((4, self.N, self.M))
            D_vec = np.zeros((4, self.N, self.M))


        AHA[0] = AHA[0] + rho_term
        AHA[3] = AHA[3] + rho_term

        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]


        for i in range(itr):


            if self.use_gpu:

                v_para = cp.fft.fft2(z_para - u_para)
                b_vec_new = [b_vec[0] + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1]),\
                             b_vec[1] + rho*(cp.conj(Dx)*v_para[2] + cp.conj(Dy)*v_para[3])]


                mu_sample = cp.real(cp.fft.ifft2((b_vec_new[0]*AHA[3] - b_vec_new[1]*AHA[1]) / determinant))
                phi_sample = cp.real(cp.fft.ifft2((b_vec_new[1]*AHA[0] - b_vec_new[0]*AHA[2]) / determinant))

                D_vec[0] = mu_sample - cp.roll(mu_sample, -1, axis=1)
                D_vec[1] = mu_sample - cp.roll(mu_sample, -1, axis=0)
                D_vec[2] = phi_sample - cp.roll(phi_sample, -1, axis=1)
                D_vec[3] = phi_sample - cp.roll(phi_sample, -1, axis=0)


                z_para = D_vec + u_para

                z_para[:2,:,:] = softTreshold(z_para[:2,:,:], lambda_u/rho, use_gpu=True)
                z_para[2:,:,:] = softTreshold(z_para[2:,:,:], lambda_p/rho, use_gpu=True)

                u_para += D_vec - z_para

                if i == itr-1:
                    mu_sample  = cp.asnumpy(mu_sample)
                    phi_sample = cp.asnumpy(phi_sample)




            else:

                v_para = fft2(z_para - u_para)
                b_vec_new = [b_vec[0] + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1]),\
                             b_vec[1] + rho*(np.conj(Dx)*v_para[2] + np.conj(Dy)*v_para[3])]


                mu_sample = np.real(ifft2((b_vec_new[0]*AHA[3] - b_vec_new[1]*AHA[1]) / determinant))
                phi_sample = np.real(ifft2((b_vec_new[1]*AHA[0] - b_vec_new[0]*AHA[2]) / determinant))

                D_vec[0] = mu_sample - np.roll(mu_sample, -1, axis=1)
                D_vec[1] = mu_sample - np.roll(mu_sample, -1, axis=0)
                D_vec[2] = phi_sample - np.roll(phi_sample, -1, axis=1)
                D_vec[3] = phi_sample - np.roll(phi_sample, -1, axis=0)


                z_para = D_vec + u_para

                z_para[:2,:,:] = softTreshold(z_para[:2,:,:], lambda_u/rho)
                z_para[2:,:,:] = softTreshold(z_para[2:,:,:], lambda_p/rho)

                u_para += D_vec - z_para

            if verbose:
                print('Number of iteration computed (%d / %d)'%(i+1,itr))
            
        return mu_sample, phi_sample
        
    
    
        
    
    def Phase_recon(self, S0_stack, method='Tikhonov', reg_u = 1e-6, reg_p = 1e-6, \
                    rho = 1e-5, lambda_u = 1e-3, lambda_p = 1e-3, itr = 20, verbose=True):
        
        
        if self.use_gpu:
            S0_stack = self.inten_normalization(cp.array(S0_stack))
            Hu = cp.array(self.Hu, copy=True)
            Hp = cp.array(self.Hp, copy=True)
            
            S0_stack_f = cp.fft.fft2(S0_stack, axes=(0,1))
            
            AHA = [cp.sum(cp.abs(Hu)**2, axis=2) + reg_u, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                   cp.sum(cp.conj(Hp)*Hu, axis=2), cp.sum(cp.abs(Hp)**2, axis=2) + reg_p]
            
            b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                     cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]
            
        else:
            S0_stack = self.inten_normalization(S0_stack)
            S0_stack_f = fft2(S0_stack,axes=(0,1))
            
            AHA = [np.sum(np.abs(self.Hu)**2, axis=2) + reg_u, np.sum(np.conj(self.Hu)*self.Hp, axis=2),\
                   np.sum(np.conj(self.Hp)*self.Hu, axis=2), np.sum(np.abs(self.Hp)**2, axis=2) + reg_p]
            
            b_vec = [np.sum(np.conj(self.Hu)*S0_stack_f, axis=2), \
                     np.sum(np.conj(self.Hp)*S0_stack_f, axis=2)]
        
        
        if method == 'Tikhonov':
            
            # Deconvolution with Tikhonov regularization
            
            mu_sample, phi_sample = self.Tikhonov_deconv_2D(AHA, b_vec)
                
                
            
        elif method == 'TV':
            
            # ADMM deconvolution with anisotropic TV regularization
            
            mu_sample, phi_sample = self.ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose)
            
        
        phi_sample -= phi_sample.mean()
        
        return mu_sample, phi_sample
    
    
    def Phase_recon_semi_2D(self, S0_stack, method='Tikhonov', reg_u = 1e-6, reg_p = 1e-6, \
                    rho = 1e-5, lambda_u = 1e-3, lambda_p = 1e-3, itr = 20, verbose=False):
        
        mu_sample = np.zeros((self.N, self.M, self.N_defocus))
        phi_sample = np.zeros((self.N, self.M, self.N_defocus))

        for i in range(self.N_defocus):
            
            if i <= self.ph_deconv_layer//2:
                tf_start_idx = self.ph_deconv_layer//2 - i
            else:
                tf_start_idx = 0

            obj_start_idx = np.maximum(0,i-self.ph_deconv_layer//2)

            if self.N_defocus -i -1 < self.ph_deconv_layer//2:
                tf_end_idx = self.ph_deconv_layer//2 + (self.N_defocus - i)
            else:
                tf_end_idx = self.ph_deconv_layer

            obj_end_idx = np.minimum(self.N_defocus,i+self.ph_deconv_layer-self.ph_deconv_layer//2)
            
            print('TF_index = (%d,%d), obj_z_index=(%d,%d), consistency: %s'\
                  %(tf_start_idx,tf_end_idx, obj_start_idx, obj_end_idx, (obj_end_idx-obj_start_idx)==(tf_end_idx-tf_start_idx)))
            
        
            if self.use_gpu:
                S0_stack_sub = self.inten_normalization(cp.array(S0_stack[:,:,obj_start_idx:obj_end_idx]))
                Hu = cp.array(self.Hu[:,:,tf_start_idx:tf_end_idx], copy=True)
                Hp = cp.array(self.Hp[:,:,tf_start_idx:tf_end_idx], copy=True)

                S0_stack_f = cp.fft.fft2(S0_stack_sub, axes=(0,1))

                AHA = [cp.sum(cp.abs(Hu)**2, axis=2) + reg_u, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                       cp.sum(cp.conj(Hp)*Hu, axis=2), cp.sum(cp.abs(Hp)**2, axis=2) + reg_p]

                b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                         cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]

            else:
                S0_stack_sub = self.inten_normalization(S0_stack[:,:,obj_start_idx:obj_end_idx])
                S0_stack_f = fft2(S0_stack_sub,axes=(0,1))
                
                Hu = self.Hu[:,:,tf_start_idx:tf_end_idx]
                Hp = self.Hp[:,:,tf_start_idx:tf_end_idx]

                AHA = [np.sum(np.abs(Hu)**2, axis=2) + reg_u, np.sum(np.conj(Hu)*self.Hp, axis=2),\
                       np.sum(np.conj(Hp)*self.Hu, axis=2), np.sum(np.abs(Hp)**2, axis=2) + reg_p]

                b_vec = [np.sum(np.conj(Hu)*S0_stack_f, axis=2), \
                         np.sum(np.conj(Hp)*S0_stack_f, axis=2)]
                
                
            if method == 'Tikhonov':

                # Deconvolution with Tikhonov regularization

                mu_sample_temp, phi_sample_temp = self.Tikhonov_deconv_2D(AHA, b_vec)



            elif method == 'TV':

                # ADMM deconvolution with anisotropic TV regularization

                mu_sample_temp, phi_sample_temp = self.ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose)


            mu_sample[:,:,i] = mu_sample_temp.copy()
            phi_sample[:,:,i] = phi_sample_temp - phi_sample_temp.mean()
            
            
        return mu_sample, phi_sample
            
        
        
        
        
        
        
    
    def Phase_recon_3D(self, S0_stack, absorption_ratio=0.0, method='Tikhonov', reg_re = 1e-4, \
                       rho = 1e-5, lambda_re = 1e-3, itr = 20, verbose=True):
        
        
        S0_stack = self.inten_normalization_3D(S0_stack)
        H_eff = self.H_re + absorption_ratio*self.H_im
        
        if self.use_gpu:
            
            S0_stack_f = cp.fft.fftn(cp.array(S0_stack.astype('float32')), axes=(0,1,2))
            H_eff = cp.array(H_eff.astype('complex64'))
            
            if method == 'Tikhonov':

                f_real = cp.asnumpy(cp.real(cp.fft.ifftn(S0_stack_f * cp.conj(H_eff) / (cp.abs(H_eff)**2 + reg_re),axes=(0,1,2))))

            if method == 'TV':

                Dx = np.zeros((self.N, self.M, self.N_defocus))
                Dy = np.zeros((self.N, self.M, self.N_defocus))
                Dz = np.zeros((self.N, self.M, self.N_defocus))
                Dx[0,0,0] = 1; Dx[0,-1,0] = -1; Dx = cp.fft.fftn(cp.array(Dx),axes=(0,1,2));
                Dy[0,0,0] = 1; Dy[-1,0,0] = -1; Dy = cp.fft.fftn(cp.array(Dy),axes=(0,1,2));
                Dz[0,0,0] = 1; Dz[0,0,-1] = -1; Dz = cp.fft.fftn(cp.array(Dz),axes=(0,1,2));

                rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy + cp.conj(Dz)*Dz)+reg_re
                AHA = cp.abs(H_eff)**2 + rho_term
                b_vec = S0_stack_f * cp.conj(H_eff)

                z_para = cp.zeros((3, self.N, self.M, self.N_defocus))
                u_para = cp.zeros((3, self.N, self.M, self.N_defocus))
                D_vec = cp.zeros((3, self.N, self.M, self.N_defocus))




                for i in range(itr):
                    v_para = cp.fft.fftn(z_para - u_para, axes=(1,2,3))
                    b_vec_new = b_vec + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1] + cp.conj(Dz)*v_para[2])


                    f_real = cp.real(cp.fft.ifftn(b_vec_new / AHA, axes=(0,1,2)))

                    D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
                    D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
                    D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)


                    z_para = D_vec + u_para

                    z_para = softTreshold(z_para, lambda_re/rho, use_gpu=True)

                    u_para += D_vec - z_para

                    if verbose:
                        print('Number of iteration computed (%d / %d)'%(i+1,itr))
                        
                    if i == itr-1:
                        f_real = cp.asnumpy(f_real)
            
        else:
        
            
            S0_stack_f = fftn(S0_stack, axes=(0,1,2))
            

            if method == 'Tikhonov':

                f_real = np.real(ifftn(S0_stack_f * np.conj(H_eff) / (np.abs(H_eff)**2 + reg_re),axes=(0,1,2)))

            if method == 'TV':

                Dx = np.zeros((self.N, self.M, self.N_defocus))
                Dy = np.zeros((self.N, self.M, self.N_defocus))
                Dz = np.zeros((self.N, self.M, self.N_defocus))
                Dx[0,0,0] = 1; Dx[0,-1,0] = -1; Dx = fftn(Dx,axes=(0,1,2));
                Dy[0,0,0] = 1; Dy[-1,0,0] = -1; Dy = fftn(Dy,axes=(0,1,2));
                Dz[0,0,0] = 1; Dz[0,0,-1] = -1; Dz = fftn(Dz,axes=(0,1,2));

                rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy + np.conj(Dz)*Dz)+reg_re
                AHA = np.abs(H_eff)**2 + rho_term
                b_vec = S0_stack_f * np.conj(H_eff)

                z_para = np.zeros((3, self.N, self.M, self.N_defocus))
                u_para = np.zeros((3, self.N, self.M, self.N_defocus))
                D_vec = np.zeros((3, self.N, self.M, self.N_defocus))




                for i in range(itr):
                    v_para = fftn(z_para - u_para, axes=(1,2,3))
                    b_vec_new = b_vec + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1] + np.conj(Dz)*v_para[2])


                    f_real = np.real(ifftn(b_vec_new / AHA, axes=(0,1,2)))

                    D_vec[0] = f_real - np.roll(f_real, -1, axis=1)
                    D_vec[1] = f_real - np.roll(f_real, -1, axis=0)
                    D_vec[2] = f_real - np.roll(f_real, -1, axis=2)


                    z_para = D_vec + u_para

                    z_para = softTreshold(z_para, lambda_re/rho)

                    u_para += D_vec - z_para

                    if verbose:
                        print('Number of iteration computed (%d / %d)'%(i+1,itr))
            
        
        # Converting to refractive index

#         n_square = self.n_media**2 *(1 - f_real*(1+1j*absorption_ratio) / (2*np.pi/self.lambda_illu)**2)
#         n_re = ((np.abs(n_square) + np.real(n_square))/2)**(0.5)
#         n_im = ((np.abs(n_square) - np.real(n_square))/2)**(0.5)
        
        return f_real*self.psz

    
    
    def Birefringence_recon(self, S1_stack, S2_stack, reg = 1e-3):
        
        # Birefringence deconvolution with slowly varying transmission approximation
        
        AHA = [np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=2) + reg, \
               np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=2), \
               -np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=2), \
               np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=2) + reg]

        S1_stack_f = fft2(S1_stack, axes=(0,1))
        S2_stack_f = fft2(S2_stack, axes=(0,1))

        b_vec = [np.sum(-np.conj(self.Hu)*S1_stack_f + np.conj(self.Hp)*S2_stack_f, axis=2), \
                 np.sum(np.conj(self.Hp)*S1_stack_f + np.conj(self.Hu)*S2_stack_f, axis=2)]

        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]

        del_phi_s = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
        del_phi_c = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))
        
        
        Retardance = 2*(del_phi_s**2 + del_phi_c**2)**(1/2)
        
        if self.cali == True:
            slowaxis = 0.5*np.arctan2(del_phi_s, -del_phi_c)%np.pi
        else:
            slowaxis = 0.5*np.arctan2(del_phi_s, del_phi_c)%np.pi
        
        
        return Retardance, slowaxis
    
