import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from PIL import Image
from IPython import display
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



def instrument_matrix_calibration(I_cali_norm, I_meas):
    
    
    
    _, N_cali = I_cali_norm.shape
    
    theta = np.r_[0:N_cali]/N_cali*2*np.pi
    S_matrix = np.array([np.ones((N_cali,)), np.cos(2*theta), np.sin(2*theta)])
    A_matrix = np.transpose(np.linalg.pinv(S_matrix.transpose()).dot(np.transpose(I_cali_norm)))
    
    if I_meas.ndim == 3:
        I_mean = np.mean(I_meas,axis=(1,2))
    elif I_meas.ndim == 4:
        I_mean = np.mean(I_meas,axis=(1,2,3))
        
    I_tot = np.sum(I_mean)
    A_matrix_S3 = I_mean/I_tot-A_matrix[:,0] 
    I_corr = (I_tot/4)*(A_matrix_S3)/np.mean(A_matrix[:,0])
    
    print('Calibrated instrument matrix:\n' + str(np.round(A_matrix,4)))
    print('Last column of instrument matrix:\n' + str(np.round(A_matrix_S3.reshape((4,1)),4)))
    
    
    plt.plot(np.transpose(I_cali_norm))
    plt.plot(np.transpose(A_matrix.dot(S_matrix)))
    plt.xlabel('Orientation of LP (deg)')
    plt.ylabel('Normalized intensity')
    plt.title('Fitted calibration curves')
    plt.legend(['$I_0$', '$I_{45}$', '$I_{90}$', '$I_{135}$'])


    return A_matrix, I_corr

class waveorder_microscopy:
    
    def __init__(self, img_dim, lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi=None,\
                 n_media=1, cali=False, bg_option='global', 
                 A_matrix=None, inc_recon=None,
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
        
        self.illumination_setup(illu_mode, NA_illu_in, Source)
                
        # select either 2D or 3D model for deconvolution
        
        self.phase_deconv_setup(phase_deconv, ph_deconv_layer)
        
        # inclination reconstruction model selection
        
        self.inclination_recon_setup(inc_recon)
                   
        # instrument matrix for polarization detection
        
        self.instrument_matrix_setup(A_matrix)
        
        
##############   constructor function group   ##############

    def illumination_setup(self, illu_mode, NA_illu_in, Source):
        
        
        if illu_mode == 'BF':
            self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
            self.N_pattern = 1
        
        elif illu_mode == 'PH':
            if NA_illu_in == None:
                raise('No inner rim NA specified in the PH illumination mode')
            else:
                self.NA_illu_in  = NA_illu_in/self.n_media
                inner_pupil = gen_Pupil(self.fxx, self.fyy, self.NA_illu_in/self.n_media, self.lambda_illu)
                self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
                self.Source -= inner_pupil
                
#                 self.Source = ifftshift(np.roll(fftshift(self.Source),(1,0),axis=(0,1)))

                
                Pupil_ring_out = gen_Pupil(self.fxx, self.fyy, self.NA_illu/self.n_media, self.lambda_illu)
                Pupil_ring_in = gen_Pupil(self.fxx, self.fyy, self.NA_illu_in/self.n_media, self.lambda_illu)
                
                
                
                self.Pupil_obj = self.Pupil_obj*np.exp((Pupil_ring_out-Pupil_ring_in)*(np.log(0.7)-1j*(np.pi/2 - 0.0*np.pi)))
                self.N_pattern = 1
                
                
        elif illu_mode == 'Arbitrary':
    
            self.Source = Source.copy()
            if Source.ndim == 2:
                self.N_pattern = 1
            else:
                self.N_pattern = len(Source)

    def phase_deconv_setup(self, phase_deconv, ph_deconv_layer):
        
        if phase_deconv == '2D':
            
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, self.z_defocus)
            self.gen_WOTF()
            
        elif phase_deconv == 'semi-2D':
            
            self.ph_deconv_layer = ph_deconv_layer
            if self.z_defocus[0] - self.z_defocus[1] >0:
                z_deconv = -(np.r_[:self.ph_deconv_layer]-self.ph_deconv_layer//2)*self.psz
            else:
                z_deconv = (np.r_[:self.ph_deconv_layer]-self.ph_deconv_layer//2)*self.psz
            
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv)
            self.G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv)
            self.gen_semi_2D_WOTF()
            
        elif phase_deconv == '3D':
            
            if self.z_defocus[0] - self.z_defocus[1] >0:
                z = -ifftshift((np.r_[0:self.N_defocus]-self.N_defocus//2)*self.psz)
            else:
                z = ifftshift((np.r_[0:self.N_defocus]-self.N_defocus//2)*self.psz)
            self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z)
            self.G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z)
            self.gen_3D_WOTF()
    
    
    def inclination_recon_setup(self, inc_recon):
        
        if inc_recon is not None:
            wave_vec_norm_x = self.lambda_illu*self.fxx
            wave_vec_norm_y = self.lambda_illu*self.fyy
            wave_vec_norm_z = (np.maximum(0,1 - wave_vec_norm_x**2 - wave_vec_norm_y**2))**(0.5)

            incident_theta = np.arctan2((wave_vec_norm_x**2 + wave_vec_norm_y**2)**(0.5), wave_vec_norm_z)
            incident_phi   = np.arctan2(wave_vec_norm_y,wave_vec_norm_x)

            if inc_recon == 'geometric':
                self.geometric_inc_matrix, self.geometric_inc_matrix_inv = gen_geometric_inc_matrix(incident_theta, incident_phi, self.Source)
                
            elif inc_recon == 'linear-diffraction-pinv' or inc_recon == 'linear-diffraction-iter':
                
                self.LD_inc_factor = np.array([np.ones_like(incident_theta), 1/2*np.cos(2*incident_theta), -1/2*np.sin(2*incident_theta)*np.cos(incident_phi), \
                                               -1/2*np.sin(2*incident_theta)*np.sin(incident_phi), -1/2*(np.sin(incident_theta)**2)*np.cos(2*incident_phi), \
                                               -1/2*(np.sin(incident_theta)**2)*np.sin(2*incident_phi)])
                self.N_inc_coeff = self.LD_inc_factor.shape[0]
                self.gen_H_OTF_inc()
                self.compute_inc_AHA()
                
    def instrument_matrix_setup(self, A_matrix):
        
        if A_matrix is None:
            self.N_channel = 5
            self.N_Stokes = 4
            self.A_matrix = 0.5*np.array([[1,0,0,-1], \
                                          [1, np.sin(self.chi), 0, -np.cos(self.chi)], \
                                          [1, 0, np.sin(self.chi), -np.cos(self.chi)], \
                                          [1, -np.sin(self.chi), 0, -np.cos(self.chi)], \
                                          [1, 0, -np.sin(self.chi), -np.cos(self.chi)]])
        else:
            self.N_channel = A_matrix.shape[0]
            self.N_Stokes = A_matrix.shape[1]
            self.A_matrix = A_matrix.copy()
        
##############   constructor asisting function group   ##############

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
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[i].copy()
            self.H_re[i], self.H_im[i] = WOTF_3D_compute(Source_current.astype('float32'), self.Pupil_obj.astype('complex64'), self.Hz_det.astype('complex64'), \
                                                         self.G_fun_z.astype('complex64'), self.psz,\
                                                         use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        self.H_re = np.squeeze(self.H_re)
        self.H_im = np.squeeze(self.H_im)
        
        
    def gen_H_OTF_inc(self):
        
        self.H_OTF_inc = np.zeros((2*self.N_inc_coeff, self.N, self.M, self.N_pattern*self.N_defocus), complex)
        
        for i,j in itertools.product(range(self.N_defocus), range(self.N_pattern)):
    
            idx = i*self.N_pattern+j
            Pupil_eff = self.Pupil_obj * self.Hz_det[:,:,i]
            
            I_norm = np.sum(self.Source[j] * Pupil_eff * Pupil_eff.conj())

            for m in range(self.N_inc_coeff):
                H1_inc = ifft2(fft2(self.Source[j] * self.LD_inc_factor[m] * Pupil_eff).conj()*fft2(Pupil_eff))
                H2_inc = ifft2(fft2(self.Source[j] * self.LD_inc_factor[m] * Pupil_eff)*fft2(Pupil_eff).conj())
                self.H_OTF_inc[2*m,:,:,idx] = (H1_inc + H2_inc)/I_norm
                self.H_OTF_inc[2*m+1,:,:,idx] = 1j*(H1_inc-H2_inc)/I_norm
                
    def compute_inc_AHA(self):
        
        self.inc_AHA = np.zeros((2*self.N_inc_coeff, 2*self.N_inc_coeff, self.N, self.M, self.N_pattern*self.N_defocus), complex)

        for i in range(2*self.N_inc_coeff):
            
            for j in range(2*self.N_inc_coeff):
                
                if i <= self.N_inc_coeff-1 and j <= self.N_inc_coeff-1:
                    self.inc_AHA[i,j] = 1/4*(self.H_OTF_inc[2*i].conj()*self.H_OTF_inc[2*j] + \
                                                self.H_OTF_inc[2*i+1].conj()*self.H_OTF_inc[2*j+1])
                    
                elif i <= self.N_inc_coeff-1 and j > self.N_inc_coeff-1:
                    self.inc_AHA[i,j] = 1/4*(-self.H_OTF_inc[2*i].conj()*self.H_OTF_inc[2*(j-self.N_inc_coeff)+1] + \
                                                self.H_OTF_inc[2*i+1].conj()*self.H_OTF_inc[2*(j-self.N_inc_coeff)])
                    
                elif i > self.N_inc_coeff-1 and j <= self.N_inc_coeff-1:
                    self.inc_AHA[i,j] = 1/4*(-self.H_OTF_inc[2*(i-self.N_inc_coeff)+1].conj()*self.H_OTF_inc[2*j] + \
                                                self.H_OTF_inc[2*(i-self.N_inc_coeff)].conj()*self.H_OTF_inc[2*j+1])
                    
                elif i > self.N_inc_coeff-1 and j > self.N_inc_coeff-1:
                    self.inc_AHA[i,j] = 1/4*(self.H_OTF_inc[2*(i-self.N_inc_coeff)+1].conj()*self.H_OTF_inc[2*(j-self.N_inc_coeff)+1] + \
                                                self.H_OTF_inc[2*(i-self.N_inc_coeff)].conj()*self.H_OTF_inc[2*(j-self.N_inc_coeff)])
        
                
    
##############   polarization computing function group   ##############

    def Stokes_recon(self, I_meas):
        
        A_pinv = np.linalg.pinv(self.A_matrix)
        Stokes_inv = lambda x:  np.transpose(np.squeeze(np.matmul(A_pinv, np.transpose(x,(1,2,0))[:,:,:,np.newaxis])),(2,0,1))
        
        
        dim = I_meas.ndim 
        
        if dim == 3:
            S_image_recon = Stokes_inv(I_meas)
        else:

            S_image_recon = np.zeros((self.N_Stokes, self.N, self.M, self.N_defocus*self.N_pattern))
            
            for i in range(self.N_defocus*self.N_pattern):
                S_image_recon[:,:,:,i] = Stokes_inv(I_meas[:,:,:,i])

            
        return S_image_recon
    
    
    def Stokes_transform(self, S_image_recon):
        
        if self.use_gpu:
            S_image_recon = cp.array(S_image_recon)
            if self.N_Stokes == 4:
                S_transformed = cp.zeros((5,)+S_image_recon.shape[1:])
            elif self.N_Stokes == 3:
                S_transformed = cp.zeros((3,)+S_image_recon.shape[1:])
        else:
            if self.N_Stokes == 4:
                S_transformed = np.zeros((5,)+S_image_recon.shape[1:])
            elif self.N_Stokes == 3:
                S_transformed = np.zeros((3,)+S_image_recon.shape[1:])
        
        S_transformed[0] = S_image_recon[0]
        
        if self.N_Stokes == 4:
            S_transformed[1] = S_image_recon[1] / S_image_recon[3]
            S_transformed[2] = S_image_recon[2] / S_image_recon[3]
            S_transformed[3] = S_image_recon[3]
            S_transformed[4] = (S_image_recon[1]**2 + S_image_recon[2]**2 + S_image_recon[3]**2)**(1/2) / S_image_recon[0] # DoP
        elif self.N_Stokes == 3:
            S_transformed[1] = S_image_recon[1] / S_image_recon[0]
            S_transformed[2] = S_image_recon[2] / S_image_recon[0]
            
        
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
            if self.N_Stokes == 4:
                S_image_tm[4] /= S_bg_tm[4]
        else:
            S_image_tm[0] /= S_bg_tm[0,:,:,np.newaxis]
            S_image_tm[1] -= S_bg_tm[1,:,:,np.newaxis]
            S_image_tm[2] -= S_bg_tm[2,:,:,np.newaxis]
            if self.N_Stokes == 4:
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
            Recon_para = cp.zeros((self.N_Stokes,)+S_image_recon.shape[1:])
        else:
            Recon_para = np.zeros((self.N_Stokes,)+S_image_recon.shape[1:])
        
        
        if self.use_gpu:
            
            if self.N_Stokes == 4:
                ret_wrapped = cp.arctan2((S_image_recon[1]**2 + S_image_recon[2]**2)**(1/2) * \
                                       S_image_recon[3], S_image_recon[3])  # retardance
            elif self.N_Stokes == 3:
                ret_wrapped = cp.arcsin(cp.minimum((S_image_recon[1]**2 + S_image_recon[2]**2)**(0.5),1))

            
            if self.cali == True:
                sa_wrapped = 0.5*cp.arctan2(-S_image_recon[1], -S_image_recon[2])%np.pi # slow-axis
            else:
                sa_wrapped = 0.5*cp.arctan2(-S_image_recon[1], S_image_recon[2])%np.pi # slow-axis
        
        else:
            
            if self.N_Stokes == 4:
                ret_wrapped = np.arctan2((S_image_recon[1]**2 + S_image_recon[2]**2)**(1/2) * \
                                           S_image_recon[3], S_image_recon[3])  # retardance
            elif self.N_Stokes == 3:
                ret_wrapped = np.arcsin(np.minimum((S_image_recon[1]**2 + S_image_recon[2]**2)**(0.5),1))
            
            
            
            if self.cali == True:
                sa_wrapped = 0.5*np.arctan2(-S_image_recon[1], -S_image_recon[2])%np.pi # slow-axis
            else:
                sa_wrapped = 0.5*np.arctan2(-S_image_recon[1], S_image_recon[2])%np.pi # slow-axis
                
        sa_wrapped[ret_wrapped<0] += np.pi/2
        ret_wrapped[ret_wrapped<0] += np.pi
        Recon_para[0] = ret_wrapped.copy()        
        Recon_para[1] = sa_wrapped%np.pi
        Recon_para[2] = S_image_recon[0] # transmittance
        
        if self.N_Stokes == 4:
            Recon_para[3] = S_image_recon[4] # DoP
        
        
        if self.use_gpu:
            Recon_para = cp.asnumpy(Recon_para)
        
        return Recon_para
    
    
    
    def Birefringence_recon(self, S1_stack, S2_stack, reg = 1e-3):
        
        # Birefringence deconvolution with slowly varying transmission approximation
        
        if self.use_gpu:
            
            
            
            Hu = cp.array(self.Hu, copy=True)
            Hp = cp.array(self.Hp, copy=True)
            
            AHA = [cp.sum(cp.abs(Hu)**2 + cp.abs(Hp)**2, axis=2) + reg, \
                   cp.sum(Hu*cp.conj(Hp) - cp.conj(Hu)*Hp, axis=2), \
                   -cp.sum(Hu*cp.conj(Hp) - cp.conj(Hu)*Hp, axis=2), \
                   cp.sum(cp.abs(Hu)**2 + cp.abs(Hp)**2, axis=2) + reg]

            S1_stack_f = cp.fft.fft2(cp.array(S1_stack), axes=(0,1))
            if self.cali:
                S2_stack_f = cp.fft.fft2(-cp.array(S2_stack), axes=(0,1))
            else:
                S2_stack_f = cp.fft.fft2(cp.array(S2_stack), axes=(0,1))

            b_vec = [cp.sum(-cp.conj(Hu)*S1_stack_f + cp.conj(Hp)*S2_stack_f, axis=2), \
                     cp.sum(cp.conj(Hp)*S1_stack_f + cp.conj(Hu)*S2_stack_f, axis=2)]
        
        else:
        
            AHA = [np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=2) + reg, \
                   np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=2), \
                   -np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=2), \
                   np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=2) + reg]

            S1_stack_f = fft2(S1_stack, axes=(0,1))
            if self.cali:
                S2_stack_f = fft2(-S2_stack, axes=(0,1))
            else:
                S2_stack_f = fft2(S2_stack, axes=(0,1))

            b_vec = [np.sum(-np.conj(self.Hu)*S1_stack_f + np.conj(self.Hp)*S2_stack_f, axis=2), \
                     np.sum(np.conj(self.Hp)*S1_stack_f + np.conj(self.Hu)*S2_stack_f, axis=2)]

    
        del_phi_s, del_phi_c = self.Tikhonov_deconv_2D(AHA, b_vec)
        
        Retardance = 2*(del_phi_s**2 + del_phi_c**2)**(1/2) 
        slowaxis = 0.5*np.arctan2(del_phi_s, del_phi_c)%np.pi
        
        
        return Retardance, slowaxis
    
    
    def Inclination_recon_geometric(self, retardance, orientation, on_axis_idx, reg_ret_ap = 1e-2):
        
        
        retardance_on_axis = retardance[:,:,on_axis_idx].copy()
        orientation_on_axis = orientation[:,:,on_axis_idx].copy()

        retardance = np.transpose(retardance,(2,0,1))
        
        N_meas = self.N_pattern * self.N_defocus
        
        
        inc_coeff = np.reshape(self.geometric_inc_matrix_inv.dot(retardance.reshape((N_meas,self.N*self.M))), (6, self.N, self.M))
        inc_coeff_sin_2theta = (inc_coeff[2]**2 + inc_coeff[3]**2)**(0.5)
        inclination = np.arctan2(retardance_on_axis*2, inc_coeff_sin_2theta)
        inclination = np.pi/2 - (np.pi/2-inclination)*np.sign(inc_coeff[2]*np.cos(orientation_on_axis)+inc_coeff[3]*np.sin(orientation_on_axis))


        retardance_ap = retardance_on_axis*np.sin(inclination)**2 / (np.sin(inclination)**4+reg_ret_ap)
        
        
        return inclination, retardance_ap, inc_coeff
    
    def Inclination_recon_LD_pinv(self, S1_stack, S2_stack, on_axis_idx, reg=1e-1*np.ones((12,)), reg_bire=1e-1, reg_ret_ap = 1e-2):
        
        
        S1_stack_f = fft2(S1_stack, axes=(0,1))
        S2_stack_f = fft2(S2_stack, axes=(0,1))
        
        
        b_vec = np.zeros((2*self.N_inc_coeff,self.N, self.M, self.N_pattern*self.N_defocus), complex)

        for i in range(2*self.N_inc_coeff):
            if i <= self.N_inc_coeff-1:
                b_vec[i] = 1/2*(-self.H_OTF_inc[2*i].conj()*S1_stack_f +\
                                self.H_OTF_inc[2*i+1].conj()*S2_stack_f)
            elif i > self.N_inc_coeff-1:
                b_vec[i] = 1/2*(self.H_OTF_inc[2*(i-self.N_inc_coeff)+1].conj()*S1_stack_f + \
                                self.H_OTF_inc[2*(i-self.N_inc_coeff)].conj()*S2_stack_f)
                
                
        # Compute on-axis deconv birefringence
        
        bire_AHA = [self.inc_AHA[0,0,:,:,on_axis_idx]+reg_bire, self.inc_AHA[0,self.N_inc_coeff,:,:,on_axis_idx], \
                   self.inc_AHA[self.N_inc_coeff,0,:,:,on_axis_idx], self.inc_AHA[self.N_inc_coeff,self.N_inc_coeff,:,:,on_axis_idx]+reg_bire]
        bire_b_vec = [b_vec[0,:,:,on_axis_idx], b_vec[self.N_inc_coeff,:,:,on_axis_idx]]
        
        del_phi_s, del_phi_c = self.Tikhonov_deconv_2D(bire_AHA, bire_b_vec)
        
        
        retardance_on_axis = (del_phi_s**2 + del_phi_c**2)**(1/2)
        orientation_on_axis = 0.5*np.arctan2(del_phi_s, del_phi_c)%np.pi
        
        AHA_energy = np.zeros((2*self.N_inc_coeff,))

        for i in range(2*self.N_inc_coeff):
            AHA_energy[i] = np.mean(np.abs(np.sum(self.inc_AHA[i,i],axis=2)))

        AHA_energy = AHA_energy/np.max(AHA_energy)
        
        inc_AHA = self.inc_AHA.copy()

        for i in range(2*self.N_inc_coeff):
            inc_AHA[i,i] += AHA_energy[i]*reg[i]

        
        
        
        # inc_coeff fit
        
        x_est = np.real(ifft2(np.transpose(np.squeeze(np.matmul(np.linalg.pinv(np.transpose(np.sum(inc_AHA,axis=4),(2,3,0,1))), 
                                 np.transpose(np.sum(b_vec,axis=3),(1,2,0))[:,:,:,np.newaxis])), (2,0,1)), axes=(1,2)))
        
        inc_coeff = ((x_est[:self.N_inc_coeff,:,:]**2+x_est[self.N_inc_coeff:,:,:]**2)**(0.5))*\
                     np.sign(x_est[:self.N_inc_coeff,:,:]*np.sin(2*orientation_on_axis) + x_est[self.N_inc_coeff:,:,:]*np.cos(2*orientation_on_axis))

        inc_coeff_sin_2theta = ((inc_coeff[2]**2 + inc_coeff[3]**2)**(0.5))


        inclination = np.arctan2(retardance_on_axis*2, inc_coeff_sin_2theta)
        inclination = np.pi/2 - (np.pi/2-inclination)*np.sign(inc_coeff[2]*np.cos(orientation_on_axis) + inc_coeff[3]*np.sin(orientation_on_axis))


        retardance_ap = retardance_on_axis*np.sin(inclination)**2 / (np.sin(inclination)**4+ reg_ret_ap)
        
        return inclination, retardance_ap, inc_coeff, retardance_on_axis, orientation_on_axis

        
    def Inclination_recon_LD_iter(self, S1_stack, S2_stack, on_axis_idx, reg=1e-1*np.ones((12,)), reg_bire=1e-1, itr=100, reg_ret_ap = 1e-2):
        
        S1_stack_f = fft2(S1_stack, axes=(0,1))
        S2_stack_f = fft2(S2_stack, axes=(0,1))
        
        
        # Compute on-axis deconv birefringence
        
        AHA = [0.25*(np.abs(self.Hu[:,:,on_axis_idx])**2 + np.abs(self.Hp[:,:,on_axis_idx])**2) + reg_bire, \
               0.25*(self.Hu[:,:,on_axis_idx]*np.conj(self.Hp[:,:,on_axis_idx]) - np.conj(self.Hu[:,:,on_axis_idx])*self.Hp[:,:,on_axis_idx]), \
               0.25*(-self.Hu[:,:,on_axis_idx]*np.conj(self.Hp[:,:,on_axis_idx]) - np.conj(self.Hu[:,:,on_axis_idx])*self.Hp[:,:,on_axis_idx]), \
               0.25*(np.abs(self.Hu[:,:,on_axis_idx])**2 + np.abs(self.Hp[:,:,on_axis_idx])**2) + reg_bire]

        b_vec = [0.5*(-np.conj(self.Hu[:,:,on_axis_idx])*S1_stack_f[:,:,on_axis_idx] + np.conj(self.Hp[:,:,on_axis_idx])*S2_stack_f[:,:,on_axis_idx]), \
                 0.5*(np.conj(self.Hp[:,:,on_axis_idx])*S1_stack_f[:,:,on_axis_idx] + np.conj(self.Hu[:,:,on_axis_idx])*S2_stack_f[:,:,on_axis_idx])]
        
        del_phi_s, del_phi_c = self.Tikhonov_deconv_2D(AHA, b_vec)
        
        retardance_on_axis = (del_phi_s**2 + del_phi_c**2)**(1/2)
        orientation_on_axis = 0.5*np.arctan2(del_phi_s, del_phi_c)%np.pi
        
        # Setup iterative algorithm parameters
        
        AHA_energy = np.zeros((2*self.N_inc_coeff,))

        for i in range(2*self.N_inc_coeff):
            AHA_energy[i] = np.mean(np.abs(np.sum(self.inc_AHA[i,i],axis=2)))

        AHA_energy = AHA_energy/np.max(AHA_energy)
        reg_norm = AHA_energy*reg
        
        err = np.zeros(itr+1)

        inc_coeff = np.zeros((self.N_inc_coeff, self.N, self.M))
        inc_coeff_sin_2theta = np.zeros((self.N, self.M))
        
        f1,ax = plt.subplots(3,2,figsize=(6,9))

        tic_time = time.time()
        
        
        # iterative estimator
        
        print('|  Iter  |  error  |  Elapsed time (sec)  |')

        for i in range(itr):

            S1_est_f = np.zeros((self.N, self.M, self.N_pattern*self.N_defocus), complex)
            S2_est_f = np.zeros((self.N, self.M, self.N_pattern*self.N_defocus), complex)
            for j in range(self.N_inc_coeff):
                S1_est_f += -0.5*self.H_OTF_inc[2*j,:,:,:]*fft2((inc_coeff[j]*np.sin(2*orientation_on_axis))[:,:,np.newaxis], axes=(0,1)) + \
                                 0.5*self.H_OTF_inc[2*j+1,:,:,:]*fft2((inc_coeff[j]*np.cos(2*orientation_on_axis))[:,:, np.newaxis], axes=(0,1)) 
                S2_est_f += 0.5*self.H_OTF_inc[2*j+1,:,:,:]*fft2((inc_coeff[j]*np.sin(2*orientation_on_axis))[:,:,np.newaxis], axes=(0,1)) + \
                                 0.5*self.H_OTF_inc[2*j,:,:,:]*fft2((inc_coeff[j]*np.cos(2*orientation_on_axis))[:,:,np.newaxis], axes=(0,1))

            S1_res = S1_stack_f - S1_est_f
            S2_res = S2_stack_f - S2_est_f

            err[i+1] = np.sum(np.abs(S1_res)**2 + np.abs(S2_res)**2)


            grad_inc_coeff = np.zeros((self.N_inc_coeff, self.N, self.M), complex)

            for j in range(self.N_inc_coeff):
                grad_inc_coeff[j] = -0.5*np.sum(ifft2(-0.5*self.H_OTF_inc[2*j,:,:,:].conj()*S1_res + 0.5*self.H_OTF_inc[2*j+1,:,:,:].conj()*S2_res, \
                                                    axes=(0,1))*np.sin(2*orientation_on_axis[:,:,np.newaxis]) + \
                                        ifft2(0.5*self.H_OTF_inc[2*j+1,:,:,:].conj()*S1_res + 0.5*self.H_OTF_inc[2*j,:,:,:].conj()*S2_res, \
                                              axes=(0,1))*np.cos(2*orientation_on_axis[:,:,np.newaxis]), axis=2) / self.N_pattern /self.N_defocus+ reg_norm[j]*inc_coeff[j]


            grad_inc_coeff_sin_2theta = 0.5*(grad_inc_coeff[2]*np.cos(orientation_on_axis) + grad_inc_coeff[3]*np.sin(orientation_on_axis))

            inc_coeff -= np.real(grad_inc_coeff)
            inc_coeff_sin_2theta -= np.real(grad_inc_coeff_sin_2theta)



            temp = inc_coeff.copy()
            temp_inc = inc_coeff_sin_2theta.copy()

            if i == 0:
                t = 1

                inc_coeff = temp.copy()
                tempp = temp.copy()

                inc_coeff_sin_2theta = temp_inc.copy()
                tempp_inc = temp_inc.copy()

            else:
                if err[i] >= err[i-1]:
                    t = 1

                    inc_coeff = temp.copy()
                    tempp = temp.copy()

                    inc_coeff_sin_2theta = temp_inc.copy()
                    tempp_inc = temp_inc.copy()

                else:
                    tp = t
                    t = (1 + (1 + 4 * tp**2)**(1/2))/2

                    inc_coeff = temp + (tp - 1) * (temp - tempp) / t
                    tempp = temp.copy()

                    inc_coeff_sin_2theta = temp_inc + (tp - 1) * (temp_inc - tempp_inc) / t
                    tempp_inc = temp_inc.copy()

            inc_coeff[2] = inc_coeff_sin_2theta*np.cos(orientation_on_axis)
            inc_coeff[3] = inc_coeff_sin_2theta*np.sin(orientation_on_axis)


            if np.mod(i,1) == 0:

                print('|  %d  |  %.2e  |   %.2f  |'%(i+1,err[i+1],time.time()-tic_time))

                if i != 0:
                    ax[0,0].cla()
                    ax[0,1].cla()
                    ax[1,0].cla()
                    ax[1,1].cla()
                    ax[2,0].cla()
                    ax[2,1].cla()




                ax[0,0].imshow(inc_coeff[0],cmap='gray')    
                ax[0,1].imshow(inc_coeff[1],cmap='gray')
                ax[1,0].imshow(inc_coeff[2],cmap='gray')    
                ax[1,1].imshow(inc_coeff[3],cmap='gray')
                ax[2,0].imshow(inc_coeff[4],cmap='gray')    
                ax[2,1].imshow(inc_coeff[5],cmap='gray')

                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)
                if i == itr-1:
                    print('|  %d  |  %.2e  |   %.2f   |'%(i+1,err[i+1],time.time()-tic_time))
                    
            
        inclination = np.arctan2(retardance_on_axis*2, inc_coeff_sin_2theta)
        retardance_ap = retardance_on_axis*np.sin(inclination)**2 / (np.sin(inclination)**4+ reg_ret_ap)
        
        return inclination, retardance_ap, inc_coeff, inc_coeff_sin_2theta, retardance_on_axis, orientation_on_axis
    
    
    
    
##############   phase computing function group   ##############

    def inten_normalization(self, S0_stack, bg_filter=True):
        
        _,_, Nimg = S0_stack.shape
        
        if self.use_gpu:
            S0_norm_stack = cp.zeros_like(S0_stack)
            
            for i in range(Nimg):
                if bg_filter:
                    S0_norm_stack[:,:,i] = S0_stack[:,:,i]/uniform_filter_2D(S0_stack[:,:,i], size=self.N//2, use_gpu=True)
                else:
                    S0_norm_stack[:,:,i] = S0_stack[:,:,i].copy()
                S0_norm_stack[:,:,i] /= S0_norm_stack[:,:,i].mean()
                S0_norm_stack[:,:,i] -= 1

        else:
            S0_norm_stack = np.zeros_like(S0_stack)
        
            for i in range(Nimg):
                if bg_filter:
                    S0_norm_stack[:,:,i] = S0_stack[:,:,i]/uniform_filter(S0_stack[:,:,i], size=self.N//2)
                else:
                    S0_norm_stack[:,:,i] = S0_stack[:,:,i].copy()
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
                    rho = 1e-5, lambda_u = 1e-3, lambda_p = 1e-3, itr = 20, verbose=True, bg_filter=True):
        
        
        if self.use_gpu:
            
            S0_stack = self.inten_normalization(cp.array(S0_stack), bg_filter=bg_filter)
            Hu = cp.array(self.Hu, copy=True)
            Hp = cp.array(self.Hp, copy=True)
            
            S0_stack_f = cp.fft.fft2(S0_stack, axes=(0,1))
            
            AHA = [cp.sum(cp.abs(Hu)**2, axis=2) + reg_u, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                   cp.sum(cp.conj(Hp)*Hu, axis=2), cp.sum(cp.abs(Hp)**2, axis=2) + reg_p]
            
            b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                     cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]
            
        else:
            S0_stack = self.inten_normalization(S0_stack, bg_filter=bg_filter)
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

                AHA = [np.sum(np.abs(Hu)**2, axis=2) + reg_u, np.sum(np.conj(Hu)*Hp, axis=2),\
                       np.sum(np.conj(Hp)*Hu, axis=2), np.sum(np.abs(Hp)**2, axis=2) + reg_p]

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
        
        return -f_real*self.psz/4/np.pi*self.lambda_illu

    
    
    
    
