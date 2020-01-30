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
                 illu_mode='BF', NA_illu_in=None, Source=None, Source_PolState=np.array([1, 1j]),
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
        self.N, self.M                 = img_dim
        self.n_media                   = n_media
        self.lambda_illu               = lambda_illu/n_media
        self.ps                        = ps
        self.z_defocus                 = z_defocus.copy()
        if len(z_defocus) >= 2:
            self.psz                   = np.abs(z_defocus[0] - z_defocus[1])
            self.G_tensor_z_upsampling = np.ceil(self.psz/(self.lambda_illu/2))
        self.NA_obj                    = NA_obj/n_media
        self.NA_illu                   = NA_illu/n_media
        self.N_defocus                 = len(z_defocus)
        self.chi                       = chi
        self.cali                      = cali
        self.bg_option                 = bg_option
             
        
        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        self.Pupil_obj = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.Pupil_support = self.Pupil_obj.copy()
        
        # illumination setup
        
        self.illumination_setup(illu_mode, NA_illu_in, Source, Source_PolState)
                
        # select either 2D or 3D model for deconvolution
        
        self.phase_deconv_setup(phase_deconv, ph_deconv_layer)
        
        # instrument matrix for polarization detection
        
        self.instrument_matrix_setup(A_matrix)
        
        # inclination reconstruction model selection
        
        self.inclination_recon_setup(inc_recon, phase_deconv)
                   
        
        
        
##############   constructor function group   ##############

    def illumination_setup(self, illu_mode, NA_illu_in, Source, Source_PolState):
        
        
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
                
            self.Source_PolState = np.zeros((self.N_pattern, 2), complex)
            
            if Source_PolState.ndim == 1:
                for i in range(self.N_pattern):
                    self.Source_PolState[i] = Source_PolState/(np.sum(np.abs(Source_PolState)**2))**(1/2)
            else:
                if len(Source_PolState) != self.N_pattern:
                    raise('The length of Source_PolState needs to be either 1 or the same as N_pattern')
                for i in range(self.N_pattern):
                    self.Source_PolState[i] = Source_PolState[i]/(np.sum(np.abs(Source_PolState[i])**2))**(1/2)
            

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
    
    
    def inclination_recon_setup(self, inc_recon, phase_deconv):
        
        if inc_recon is not None and inc_recon != '3D':
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
            elif inc_recon == '2D-vec-WOTF' and phase_deconv == '2D':
                self.gen_2D_vec_WOTF_inc()
                self.inc_AHA_2D_vec = np.zeros((7,7,self.N,self.M),complex)

                for i,j,p in itertools.product(range(7), range(7), range(self.N_Stokes)):
                    self.inc_AHA_2D_vec[i,j] += np.sum(np.conj(self.H_dyadic_2D_OTF[p,i])*self.H_dyadic_2D_OTF[p,j],axis=2)

                
        elif inc_recon == '3D' and phase_deconv == '3D':
            self.gen_3D_vec_WOTF_inc()
            self.inc_AHA_3D_vec = np.zeros((7,7,self.N,self.M,self.N_defocus), dtype='complex64')

            for i,j,p in itertools.product(range(7), range(7), range(self.N_Stokes)):
                self.inc_AHA_3D_vec[i,j] += np.sum(np.conj(self.H_dyadic_OTF[p,i])*self.H_dyadic_OTF[p,j],axis=0)
            
                
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
        
        
        for i,j in itertools.product(range(self.ph_deconv_layer), range(self.N_pattern)):

            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[j].copy()

            idx = i*self.N_pattern+j
            self.Hu[:,:,idx], self.Hp[:,:,idx] = WOTF_semi_2D_compute(Source_current, Source_current, self.Pupil_obj, self.Hz_det[:,:,i], \
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
            self.H_re[i], self.H_im[i] = WOTF_3D_compute(Source_current.astype('float32'), Source_current.astype('float32'), self.Pupil_obj.astype('complex64'), \
                                                         self.Hz_det.astype('complex64'),  self.G_fun_z.astype('complex64'), self.psz,\
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
                    
    def gen_3D_WOTF_inc(self):
        
        
        self.H_plus_re   = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_plus_im   = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_minus_re  = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_minus_im  = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_Cplus_re  = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_Cplus_im  = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_Cminus_re = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_inc_1     = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        self.H_inc_2     = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        
        
        illumination_factor_plus   = (1 - 0.5*(self.lambda_illu**2)*(self.fxx**2 + self.fyy**2))*self.Pupil_support
        illumination_factor_minus  = (self.lambda_illu**2)*(self.fyy**2 - self.fxx**2)/2
        illumination_factor_Cplus  = -(self.lambda_illu**2)*self.fxx*self.fyy
        illumination_factor_Cminus = (1-(self.lambda_illu**2)*(self.fxx**2 + self.fyy**2)*self.Pupil_support)**(0.5)*self.Pupil_support
        illumination_factor_inc_1  = -self.lambda_illu*(self.fxx*illumination_factor_Cminus - 1j*self.fyy)/2
        illumination_factor_inc_2  = -self.lambda_illu*(1j*self.fxx + self.fyy*illumination_factor_Cminus)/2
        
        OTF_compute = lambda x, y: WOTF_3D_compute(x.astype('float32'), (x*y).astype('complex64'), 
                                                 self.Pupil_obj.astype('complex64'), self.Hz_det.astype('complex64'), \
                                                 self.G_fun_z.astype('complex64'), self.psz,\
                                                 use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[i].copy()
            
            self.H_plus_re[i],  self.H_plus_im[i]  = OTF_compute(Source_current,illumination_factor_plus)
            self.H_minus_re[i], self.H_minus_im[i] = OTF_compute(Source_current,illumination_factor_minus)
            self.H_Cplus_re[i], self.H_Cplus_im[i] = OTF_compute(Source_current,illumination_factor_Cplus)
            
            _, self.H_Cminus_re[i] = OTF_compute(Source_current,illumination_factor_Cminus)            
            self.H_inc_1[i], _     = OTF_compute(Source_current,illumination_factor_inc_1)
            self.H_inc_2[i], _     = OTF_compute(Source_current,illumination_factor_inc_2)
            
    def gen_2D_vec_WOTF_inc(self):
        
        
        self.H_dyadic_2D_OTF = np.zeros((self.N_Stokes, 7, self.N, self.M, self.N_defocus*self.N_pattern),dtype='complex64')

        
        
        fr = (self.fxx**2 + self.fyy**2)**(0.5)
        cos_factor = (1-(self.lambda_illu**2)*(fr**2)*self.Pupil_support)**(0.5)*self.Pupil_support
        dc_idx = (fr==0)
        nondc_idx = (fr!=0)
        E_field_factor = np.zeros((5, self.N, self.M))
        
        E_field_factor[0, nondc_idx] = ((self.fxx[nondc_idx]**2)*cos_factor[nondc_idx]+ self.fyy[nondc_idx]**2) / fr[nondc_idx]**2
        E_field_factor[0, dc_idx] = 1
        E_field_factor[1, nondc_idx] = (self.fxx[nondc_idx]*self.fyy[nondc_idx] * (cos_factor[nondc_idx]-1)) / fr[nondc_idx]**2
        E_field_factor[2, nondc_idx] = ((self.fyy[nondc_idx]**2)*cos_factor[nondc_idx] + self.fxx[nondc_idx]**2) / fr[nondc_idx]**2
        E_field_factor[2, dc_idx] = 1
        E_field_factor[3, nondc_idx] = -self.lambda_illu*self.fxx[nondc_idx]
        E_field_factor[4, nondc_idx] = -self.lambda_illu*self.fyy[nondc_idx]
        
        G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, self.z_defocus)
        G_tensor_z = gen_dyadic_Greens_tensor_z(self.fxx, self.fyy, G_fun_z, self.Pupil_support, self.lambda_illu)


        OTF_compute = lambda x, y, z, w: WOTF_semi_2D_compute(x, y, self.Pupil_obj, w, \
                                                           z, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        

        for i,j in itertools.product(range(self.N_defocus), range(self.N_pattern)):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[j].copy()
            
            idx = i*self.N_pattern+j
            
            Ex_field = self.Source_PolState[j,0]*E_field_factor[0] + self.Source_PolState[j,1]*E_field_factor[1]
            Ey_field = self.Source_PolState[j,0]*E_field_factor[1] + self.Source_PolState[j,1]*E_field_factor[2]
            Ez_field = self.Source_PolState[j,0]*E_field_factor[3] + self.Source_PolState[j,1]*E_field_factor[4]
            
            IF_ExEx = np.abs(Ex_field)**2
            IF_ExEy = Ex_field * np.conj(Ey_field)
            IF_ExEz = Ex_field * np.conj(Ez_field)
            IF_EyEy = np.abs(Ey_field)**2
            IF_EyEz = Ey_field * np.conj(Ez_field)

            Source_norm = Source_current*(IF_ExEx + IF_EyEy)

            ExEx_Gxx_re, ExEx_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            ExEy_Gxy_re, ExEy_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            ExEz_Gxz_re, ExEz_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            EyEx_Gyx_re, EyEx_Gyx_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            EyEy_Gyy_re, EyEy_Gyy_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            EyEz_Gyz_re, EyEz_Gyz_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])
            ExEx_Gxy_re, ExEx_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            ExEy_Gxx_re, ExEy_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            EyEx_Gyy_re, EyEx_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            EyEy_Gyx_re, EyEy_Gyx_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            ExEx_Gxz_re, ExEx_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            ExEz_Gxx_re, ExEz_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            EyEx_Gyz_re, EyEx_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])
            EyEz_Gyx_re, EyEz_Gyx_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            ExEy_Gxz_re, ExEy_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            ExEz_Gxy_re, ExEz_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,1,:,:,i], self.Hz_det[:,:,i])
            EyEy_Gyz_re, EyEy_Gyz_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])
            EyEz_Gyy_re, EyEz_Gyy_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            ExEy_Gyy_re, ExEy_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            ExEz_Gyz_re, ExEz_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])
            EyEx_Gxx_re, EyEx_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            EyEz_Gxz_re, EyEz_Gxz_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            ExEx_Gyy_re, ExEx_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            EyEy_Gxx_re, EyEy_Gxx_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            EyEx_Gxz_re, EyEx_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            EyEz_Gxx_re, EyEz_Gxx_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,0,:,:,i], self.Hz_det[:,:,i])
            ExEy_Gyz_re, ExEy_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])
            ExEz_Gyy_re, ExEz_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[1,1,:,:,i], self.Hz_det[:,:,i])
            EyEy_Gxz_re, EyEy_Gxz_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,2,:,:,i], self.Hz_det[:,:,i])
            ExEx_Gyz_re, ExEx_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[1,2,:,:,i], self.Hz_det[:,:,i])


            self.H_dyadic_2D_OTF[0,0,:,:,idx] = ExEx_Gxx_re + ExEy_Gxy_re + ExEz_Gxz_re + EyEx_Gyx_re + EyEy_Gyy_re + EyEz_Gyz_re
            self.H_dyadic_2D_OTF[0,1,:,:,idx] = ExEx_Gxx_im + ExEy_Gxy_im + ExEz_Gxz_im + EyEx_Gyx_im + EyEy_Gyy_im + EyEz_Gyz_im
            self.H_dyadic_2D_OTF[0,2,:,:,idx] = ExEx_Gxx_re - ExEy_Gxy_re + EyEx_Gyx_re - EyEy_Gyy_re
            self.H_dyadic_2D_OTF[0,3,:,:,idx] = ExEx_Gxy_re + ExEy_Gxx_re + EyEx_Gyy_re + EyEy_Gyx_re
            self.H_dyadic_2D_OTF[0,4,:,:,idx] = ExEx_Gxz_re + ExEz_Gxx_re + EyEx_Gyz_re + EyEz_Gyx_re
            self.H_dyadic_2D_OTF[0,5,:,:,idx] = ExEy_Gxz_re + ExEz_Gxy_re + EyEy_Gyz_re + EyEz_Gyy_re
            self.H_dyadic_2D_OTF[0,6,:,:,idx] = ExEz_Gxz_re + EyEz_Gyz_re

            self.H_dyadic_2D_OTF[1,0,:,:,idx] = ExEx_Gxx_re + ExEy_Gxy_re + ExEz_Gxz_re - EyEx_Gyx_re - EyEy_Gyy_re - EyEz_Gyz_re
            self.H_dyadic_2D_OTF[1,1,:,:,idx] = ExEx_Gxx_im + ExEy_Gxy_im + ExEz_Gxz_im - EyEx_Gyx_im - EyEy_Gyy_im - EyEz_Gyz_im
            self.H_dyadic_2D_OTF[1,2,:,:,idx] = ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
            self.H_dyadic_2D_OTF[1,3,:,:,idx] = ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
            self.H_dyadic_2D_OTF[1,4,:,:,idx] = ExEx_Gxz_re + ExEz_Gxx_re - EyEx_Gyz_re - EyEz_Gyx_re
            self.H_dyadic_2D_OTF[1,5,:,:,idx] = ExEy_Gxz_re + ExEz_Gxy_re - EyEy_Gyz_re - EyEz_Gyy_re
            self.H_dyadic_2D_OTF[1,6,:,:,idx] = ExEz_Gxz_re - EyEz_Gyz_re

            self.H_dyadic_2D_OTF[2,0,:,:,idx] = ExEx_Gxy_re + ExEy_Gyy_re + ExEz_Gyz_re + EyEx_Gxx_re + EyEy_Gyx_re + EyEz_Gxz_re
            self.H_dyadic_2D_OTF[2,1,:,:,idx] = ExEx_Gxy_im + ExEy_Gyy_im + ExEz_Gyz_im + EyEx_Gxx_im + EyEy_Gyx_im + EyEz_Gxz_im
            self.H_dyadic_2D_OTF[2,2,:,:,idx] = ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
            self.H_dyadic_2D_OTF[2,3,:,:,idx] = ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
            self.H_dyadic_2D_OTF[2,4,:,:,idx] = ExEx_Gyz_re + ExEz_Gxy_re + EyEx_Gxz_re + EyEz_Gxx_re
            self.H_dyadic_2D_OTF[2,5,:,:,idx] = ExEy_Gyz_re + ExEz_Gyy_re + EyEy_Gxz_re + EyEz_Gyx_re
            self.H_dyadic_2D_OTF[2,6,:,:,idx] = ExEz_Gyz_re + EyEz_Gxz_re
            
            if self.N_Stokes == 4:
        
                self.H_dyadic_2D_OTF[3,0,:,:,idx] = -ExEx_Gxy_im - ExEy_Gyy_im - ExEz_Gyz_im + EyEx_Gxx_im + EyEy_Gyx_im + EyEz_Gxz_im
                self.H_dyadic_2D_OTF[3,1,:,:,idx] =  ExEx_Gxy_re + ExEy_Gyy_re + ExEz_Gyz_re - EyEx_Gxx_re - EyEy_Gyx_re - EyEz_Gxz_re
                self.H_dyadic_2D_OTF[3,2,:,:,idx] = -ExEx_Gxy_im + ExEy_Gyy_im + EyEx_Gxx_im - EyEy_Gyx_im
                self.H_dyadic_2D_OTF[3,3,:,:,idx] = -ExEx_Gyy_im - ExEy_Gxy_im + EyEx_Gyx_im + EyEy_Gxx_im
                self.H_dyadic_2D_OTF[3,4,:,:,idx] = -ExEx_Gyz_im - ExEz_Gxy_im + EyEx_Gxz_im + EyEz_Gxx_im
                self.H_dyadic_2D_OTF[3,5,:,:,idx] = -ExEy_Gyz_im - ExEz_Gyy_im + EyEy_Gxz_im + EyEz_Gyx_im
                self.H_dyadic_2D_OTF[3,6,:,:,idx] = -ExEz_Gyz_im + EyEz_Gxz_im
            
    def gen_3D_vec_WOTF_inc(self):
        
        
        self.H_dyadic_OTF = np.zeros((self.N_Stokes, 7, self.N_pattern, self.N, self.M, self.N_defocus),dtype='complex64')
        
        fr = (self.fxx**2 + self.fyy**2)**(0.5)
        cos_factor = (1-(self.lambda_illu**2)*(fr**2)*self.Pupil_support)**(0.5)*self.Pupil_support
        dc_idx = (fr==0)
        nondc_idx = (fr!=0)
        E_field_factor = np.zeros((5, self.N, self.M))
        
        E_field_factor[0, nondc_idx] = ((self.fxx[nondc_idx]**2)*cos_factor[nondc_idx]+ self.fyy[nondc_idx]**2) / fr[nondc_idx]**2
        E_field_factor[0, dc_idx] = 1
        E_field_factor[1, nondc_idx] = (self.fxx[nondc_idx]*self.fyy[nondc_idx] * (cos_factor[nondc_idx]-1)) / fr[nondc_idx]**2
        E_field_factor[2, nondc_idx] = ((self.fyy[nondc_idx]**2)*cos_factor[nondc_idx] + self.fxx[nondc_idx]**2) / fr[nondc_idx]**2
        E_field_factor[2, dc_idx] = 1
        E_field_factor[3, nondc_idx] = -self.lambda_illu*self.fxx[nondc_idx]
        E_field_factor[4, nondc_idx] = -self.lambda_illu*self.fyy[nondc_idx]
        

        

        N_defocus = self.G_tensor_z_upsampling*self.N_defocus
        psz = self.psz/self.G_tensor_z_upsampling
        if self.z_defocus[0] - self.z_defocus[1] >0:
            z = -ifftshift((np.r_[0:N_defocus]-N_defocus//2)*psz)
        else:
            z = ifftshift((np.r_[0:N_defocus]-N_defocus//2)*psz)
        G_fun_z = gen_Greens_function_z(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z)

        G_real = fftshift(ifft2(G_fun_z, axes=(0,1))/self.ps**2)
        G_tensor = gen_dyadic_Greens_tensor(G_real, self.ps, psz, self.lambda_illu, space='Fourier')
        G_tensor_z = (ifft(G_tensor, axis=4)/psz)[...,::np.int(self.G_tensor_z_upsampling)]
        


        OTF_compute = lambda x, y, z: WOTF_3D_compute(x.astype('float32'), y.astype('complex64'), 
                                                      self.Pupil_obj.astype('complex64'), self.Hz_det.astype('complex64'), \
                                                      z.astype('complex64'), self.psz,\
                                                      use_gpu=self.use_gpu, gpu_id=self.gpu_id)

        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                Source_current = self.Source.copy()
            else:
                Source_current = self.Source[i].copy()
                
            Ex_field = self.Source_PolState[i,0]*E_field_factor[0] + self.Source_PolState[i,1]*E_field_factor[1]
            Ey_field = self.Source_PolState[i,0]*E_field_factor[1] + self.Source_PolState[i,1]*E_field_factor[2]
            Ez_field = self.Source_PolState[i,0]*E_field_factor[3] + self.Source_PolState[i,1]*E_field_factor[4]
            
            IF_ExEx = np.abs(Ex_field)**2
            IF_ExEy = Ex_field * np.conj(Ey_field)
            IF_ExEz = Ex_field * np.conj(Ez_field)
            IF_EyEy = np.abs(Ey_field)**2
            IF_EyEz = Ey_field * np.conj(Ez_field)
            
            Source_norm = Source_current*(IF_ExEx + IF_EyEy)

            ExEx_Gxx_re, ExEx_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,0])
            ExEy_Gxy_re, ExEy_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,1])
            ExEz_Gxz_re, ExEz_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,2])
            EyEx_Gyx_re, EyEx_Gyx_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,1])
            EyEy_Gyy_re, EyEy_Gyy_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[1,1])
            EyEz_Gyz_re, EyEz_Gyz_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[1,2])
            ExEx_Gxy_re, ExEx_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,1])
            ExEy_Gxx_re, ExEy_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,0])
            EyEx_Gyy_re, EyEx_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[1,1])
            EyEy_Gyx_re, EyEy_Gyx_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,1])
            ExEx_Gxz_re, ExEx_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[0,2])
            ExEz_Gxx_re, ExEz_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,0])
            EyEx_Gyz_re, EyEx_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[1,2])
            EyEz_Gyx_re, EyEz_Gyx_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,1])
            ExEy_Gxz_re, ExEy_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[0,2])
            ExEz_Gxy_re, ExEz_Gxy_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[0,1])
            EyEy_Gyz_re, EyEy_Gyz_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[1,2])
            EyEz_Gyy_re, EyEz_Gyy_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[1,1])
            ExEy_Gyy_re, ExEy_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[1,1])
            ExEz_Gyz_re, ExEz_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[1,2])
            EyEx_Gxx_re, EyEx_Gxx_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,0])
            EyEz_Gxz_re, EyEz_Gxz_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,2])
            ExEx_Gyy_re, ExEx_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[1,1])
            EyEy_Gxx_re, EyEy_Gxx_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,0])
            EyEx_Gxz_re, EyEx_Gxz_im = OTF_compute(Source_norm, Source_current*IF_ExEy.conj(), G_tensor_z[0,2])
            EyEz_Gxx_re, EyEz_Gxx_im = OTF_compute(Source_norm, Source_current*IF_EyEz, G_tensor_z[0,0])
            ExEy_Gyz_re, ExEy_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEy, G_tensor_z[1,2])
            ExEz_Gyy_re, ExEz_Gyy_im = OTF_compute(Source_norm, Source_current*IF_ExEz, G_tensor_z[1,1])
            EyEy_Gxz_re, EyEy_Gxz_im = OTF_compute(Source_norm, Source_current*IF_EyEy, G_tensor_z[0,2])
            ExEx_Gyz_re, ExEx_Gyz_im = OTF_compute(Source_norm, Source_current*IF_ExEx, G_tensor_z[1,2])


            self.H_dyadic_OTF[0,0,i] = ExEx_Gxx_re + ExEy_Gxy_re + ExEz_Gxz_re + EyEx_Gyx_re + EyEy_Gyy_re + EyEz_Gyz_re
            self.H_dyadic_OTF[0,1,i] = ExEx_Gxx_im + ExEy_Gxy_im + ExEz_Gxz_im + EyEx_Gyx_im + EyEy_Gyy_im + EyEz_Gyz_im
            self.H_dyadic_OTF[0,2,i] = ExEx_Gxx_re - ExEy_Gxy_re + EyEx_Gyx_re - EyEy_Gyy_re
            self.H_dyadic_OTF[0,3,i] = ExEx_Gxy_re + ExEy_Gxx_re + EyEx_Gyy_re + EyEy_Gyx_re
            self.H_dyadic_OTF[0,4,i] = ExEx_Gxz_re + ExEz_Gxx_re + EyEx_Gyz_re + EyEz_Gyx_re
            self.H_dyadic_OTF[0,5,i] = ExEy_Gxz_re + ExEz_Gxy_re + EyEy_Gyz_re + EyEz_Gyy_re
            self.H_dyadic_OTF[0,6,i] = ExEz_Gxz_re + EyEz_Gyz_re

            self.H_dyadic_OTF[1,0,i] = ExEx_Gxx_re + ExEy_Gxy_re + ExEz_Gxz_re - EyEx_Gyx_re - EyEy_Gyy_re - EyEz_Gyz_re
            self.H_dyadic_OTF[1,1,i] = ExEx_Gxx_im + ExEy_Gxy_im + ExEz_Gxz_im - EyEx_Gyx_im - EyEy_Gyy_im - EyEz_Gyz_im
            self.H_dyadic_OTF[1,2,i] = ExEx_Gxx_re - ExEy_Gxy_re - EyEx_Gyx_re + EyEy_Gyy_re
            self.H_dyadic_OTF[1,3,i] = ExEx_Gxy_re + ExEy_Gxx_re - EyEx_Gyy_re - EyEy_Gyx_re
            self.H_dyadic_OTF[1,4,i] = ExEx_Gxz_re + ExEz_Gxx_re - EyEx_Gyz_re - EyEz_Gyx_re
            self.H_dyadic_OTF[1,5,i] = ExEy_Gxz_re + ExEz_Gxy_re - EyEy_Gyz_re - EyEz_Gyy_re
            self.H_dyadic_OTF[1,6,i] = ExEz_Gxz_re - EyEz_Gyz_re

            self.H_dyadic_OTF[2,0,i] = ExEx_Gxy_re + ExEy_Gyy_re + ExEz_Gyz_re + EyEx_Gxx_re + EyEy_Gyx_re + EyEz_Gxz_re
            self.H_dyadic_OTF[2,1,i] = ExEx_Gxy_im + ExEy_Gyy_im + ExEz_Gyz_im + EyEx_Gxx_im + EyEy_Gyx_im + EyEz_Gxz_im
            self.H_dyadic_OTF[2,2,i] = ExEx_Gxy_re - ExEy_Gyy_re + EyEx_Gxx_re - EyEy_Gyx_re
            self.H_dyadic_OTF[2,3,i] = ExEx_Gyy_re + ExEy_Gxy_re + EyEx_Gyx_re + EyEy_Gxx_re
            self.H_dyadic_OTF[2,4,i] = ExEx_Gyz_re + ExEz_Gxy_re + EyEx_Gxz_re + EyEz_Gxx_re
            self.H_dyadic_OTF[2,5,i] = ExEy_Gyz_re + ExEz_Gyy_re + EyEy_Gxz_re + EyEz_Gyx_re
            self.H_dyadic_OTF[2,6,i] = ExEz_Gyz_re + EyEz_Gxz_re
            
            if self.N_Stokes == 4:
        
                self.H_dyadic_OTF[3,0,i] = -ExEx_Gxy_im - ExEy_Gyy_im - ExEz_Gyz_im + EyEx_Gxx_im + EyEy_Gyx_im + EyEz_Gxz_im
                self.H_dyadic_OTF[3,1,i] =  ExEx_Gxy_re + ExEy_Gyy_re + ExEz_Gyz_re - EyEx_Gxx_re - EyEy_Gyx_re - EyEz_Gxz_re
                self.H_dyadic_OTF[3,2,i] = -ExEx_Gxy_im + ExEy_Gyy_im + EyEx_Gxx_im - EyEy_Gyx_im
                self.H_dyadic_OTF[3,3,i] = -ExEx_Gyy_im - ExEy_Gxy_im + EyEx_Gyx_im + EyEy_Gxx_im
                self.H_dyadic_OTF[3,4,i] = -ExEx_Gyz_im - ExEz_Gxy_im + EyEx_Gxz_im + EyEz_Gxx_im
                self.H_dyadic_OTF[3,5,i] = -ExEy_Gyz_im - ExEz_Gyy_im + EyEy_Gxz_im + EyEz_Gyx_im
                self.H_dyadic_OTF[3,6,i] = -ExEz_Gyz_im + EyEz_Gxz_im
        
            
                
    
##############   polarization computing function group   ##############

    def Stokes_recon(self, I_meas):
        
        img_shape = I_meas.shape
        dim = I_meas.ndim 
        
        A_pinv = np.linalg.pinv(self.A_matrix)
        S_image_recon = np.reshape(np.dot(A_pinv, I_meas.reshape((self.N_channel, -1))), (self.N_Stokes,)+img_shape[1:])

            
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
                S_image_tm[1] -= uniform_filter_2D(S_image_tm[1], size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                S_image_tm[2] -= uniform_filter_2D(S_image_tm[2], size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            else:
                if self.use_gpu:
                    S1_bg = uniform_filter_2D(cp.mean(S_image_tm[1],axis=-1), size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                    S2_bg = uniform_filter_2D(cp.mean(S_image_tm[2],axis=-1), size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                else:
                    S1_bg = uniform_filter_2D(np.mean(S_image_tm[1],axis=-1), size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                    S2_bg = uniform_filter_2D(np.mean(S_image_tm[2],axis=-1), size=kernel_size, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                    
                
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

    
        del_phi_s, del_phi_c = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
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
        
        del_phi_s, del_phi_c = Dual_variable_Tikhonov_deconv_2D(bire_AHA, bire_b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        
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
        
        del_phi_s, del_phi_c = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
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
    
    def scattering_potential_tensor_recon_2D_vec(self, S_image_recon, reg_inc=1e-1*np.ones((7,))):
        
        
        start_time = time.time()

        S_stack_f = fft2(S_image_recon, axes=(1,2))
        
        AHA = self.inc_AHA_2D_vec.copy()
        
        for i in range(7):
            AHA[i,i] += np.mean(np.abs(AHA[i,i]))*reg_inc[i]

        b_vec = np.zeros((7,self.N,self.M), complex)

        for i,j in itertools.product(range(7), range(self.N_Stokes)):
            b_vec[i] += np.sum(np.conj(self.H_dyadic_2D_OTF[j,i])*S_stack_f[j],axis=2)

        
        print('Finished preprocess, elapsed time: %.2f'%(time.time()-start_time))
        
        if self.use_gpu:
        
            AHA = cp.array(AHA)
            b_vec = cp.array(b_vec)

            determinant = array_based_7x7_det(AHA)

            f_tensor = cp.zeros((7, self.N, self.M))

            for i in range(7):
                AHA_b_vec = AHA.copy()
                AHA_b_vec[:,i] = b_vec.copy()
                f_tensor[i] = cp.real(cp.fft.ifft2(array_based_7x7_det(AHA_b_vec) / determinant))

            f_tensor = cp.asnumpy(f_tensor)

        else:
            
            AHA_pinv = np.linalg.pinv(np.transpose(AHA,(2,3,0,1)))
            f_tensor = np.real(ifft2(np.transpose(np.squeeze(np.matmul(AHA_pinv, np.transpose(b_vec,(1,2,0))[...,np.newaxis])),(2,0,1)),axes=(1,2)))
            
        
        print('Finished reconstruction, elapsed time: %.2f'%(time.time()-start_time))
        
        return f_tensor
    
    
    
    def scattering_potential_tensor_recon_3D_vec(self, S_image_recon, reg_inc=1e-1*np.ones((7,))):
        
        
        start_time = time.time()
        
        S_stack_f = fftn(S_image_recon,axes=(-3,-2,-1))


        AHA = self.inc_AHA_3D_vec.copy()
        
        for i in range(7):
            AHA[i,i] += np.mean(np.abs(AHA[i,i]))*reg_inc[i]

        b_vec = np.zeros((7,self.N,self.M,self.N_defocus), dtype='complex64')

        for i,j in itertools.product(range(7), range(self.N_Stokes)):
            b_vec[i] += np.sum(np.conj(self.H_dyadic_OTF[j,i])*S_stack_f[j],axis=0)
        
        print('Finished preprocess, elapsed time: %.2f'%(time.time()-start_time))
        
        if self.use_gpu:
        
            AHA = cp.array(AHA)
            b_vec = cp.array(b_vec)

            determinant = array_based_7x7_det(AHA)

            f_tensor = cp.zeros((7, self.N,self.M,self.N_defocus), dtype='float32')

            for i in range(7):
                AHA_b_vec = AHA.copy()
                AHA_b_vec[:,i] = b_vec.copy()
                f_tensor[i] = cp.real(cp.fft.ifftn(array_based_7x7_det(AHA_b_vec) / determinant))

            f_tensor = cp.asnumpy(f_tensor)

        else:
            
            AHA_pinv = np.linalg.pinv(np.transpose(AHA,(2,3,4,0,1)))
            f_tensor = np.real(ifftn(np.transpose(np.squeeze(np.matmul(AHA_pinv, np.transpose(b_vec,(1,2,3,0))[...,np.newaxis])),(3,0,1,2)),axes=(1,2,3)))
            
        
        print('Finished reconstruction, elapsed time: %.2f'%(time.time()-start_time))
        
        return f_tensor
    
    def scattering_potential_tensor_to_3D_orientation(self, f_tensor, S_image_recon, material_type='positive', reg_ret_ap = 1e-2, full_fitting=False, itr=20):
        
        if material_type == 'positive' or 'unknown':
            
            # Positive uniaxial material
            
            azimuth_p = (np.arctan2(-f_tensor[3], -f_tensor[2])/2)%np.pi
            del_f_sin_square_p = -f_tensor[2]*np.cos(2*azimuth_p) - f_tensor[3]*np.sin(2*azimuth_p)
            del_f_sin2theta_p = -f_tensor[4]*np.cos(azimuth_p) - f_tensor[5]*np.sin(azimuth_p)
            theta_p = np.arctan2(2*del_f_sin_square_p, del_f_sin2theta_p)
            retardance_ap_p = del_f_sin_square_p * np.sin(theta_p)**2 / (np.sin(theta_p)**4 + reg_ret_ap)
            
            if material_type == 'positive':
                
                return retardance_ap_p, azimuth_p, theta_p
            
        if material_type == 'negative' or 'unknown':
            
            # Negative uniaxial material

            azimuth_n = (np.arctan2(f_tensor[3], f_tensor[2])/2)%np.pi
            del_f_sin_square_n = f_tensor[2]*np.cos(2*azimuth_n) + f_tensor[3]*np.sin(2*azimuth_n)
            del_f_sin2theta_n = f_tensor[4]*np.cos(azimuth_n) + f_tensor[5]*np.sin(azimuth_n)
            theta_n = np.arctan2(2*del_f_sin_square_n, del_f_sin2theta_n)
            retardance_ap_n = -del_f_sin_square_n * np.sin(theta_n)**2 / (np.sin(theta_n)**4 + reg_ret_ap)
            
            if material_type == 'negative':
                
                return retardance_ap_n, azimuth_n, theta_n
            
        
        if material_type == 'unknown':
            
            if f_tensor.ndim == 4:
                S_stack_f = fftn(S_image_recon,axes=(-3,-2,-1))
                
            elif f_tensor.ndim == 3:
                S_stack_f = fft2(S_image_recon,axes=(1,2))
                
            f_tensor_p = np.zeros((5,)+f_tensor.shape[1:])
            f_tensor_p[0] = -retardance_ap_p*(np.sin(theta_p)**2)*np.cos(2*azimuth_p)
            f_tensor_p[1] = -retardance_ap_p*(np.sin(theta_p)**2)*np.sin(2*azimuth_p)
            f_tensor_p[2] = -retardance_ap_p*(np.sin(2*theta_p))*np.cos(azimuth_p)
            f_tensor_p[3] = -retardance_ap_p*(np.sin(2*theta_p))*np.sin(azimuth_p)
            
            if full_fitting == True:
                f_tensor_p[4] = retardance_ap_p*(np.sin(theta_p)**2 - 2*np.cos(theta_p)**2)

            f_tensor_n = np.zeros((5,)+f_tensor.shape[1:])
            f_tensor_n[0] = -retardance_ap_n*(np.sin(theta_n)**2)*np.cos(2*azimuth_n)
            f_tensor_n[1] = -retardance_ap_n*(np.sin(theta_n)**2)*np.sin(2*azimuth_n)
            f_tensor_n[2] = -retardance_ap_n*(np.sin(2*theta_n))*np.cos(azimuth_n)
            f_tensor_n[3] = -retardance_ap_n*(np.sin(2*theta_n))*np.sin(azimuth_n)
            
            if full_fitting == True:
                f_tensor_n[4] = retardance_ap_n*(np.sin(theta_n)**2 - 2*np.cos(theta_n)**2)

            f_vec  = f_tensor.copy()

            x_map = np.zeros(f_tensor.shape[1:])
            p_map = 1/(1+np.exp(-x_map))

            err = np.zeros(itr+1)

            tic_time = time.time()
            print('|  Iter  |  error  |  Elapsed time (sec)  |')
            
            f1,ax = plt.subplots(1,2,figsize=(20,10))
            
            
            for i in range(itr):
    
                p_map = 1/(1+np.exp(-x_map))

                for j in range(4):
                    f_vec[j+2] = p_map*f_tensor_p[j] + (1-p_map)*f_tensor_n[j]
                
                if full_fitting == True:
                    f_vec[6] = p_map*f_tensor_p[4] + (1-p_map)*f_tensor_n[4]
                
                if f_tensor.ndim == 4:
                    f_vec_f = fftn(f_vec, axes=(1,2,3))
                    S_est_vec = np.zeros((self.N_Stokes, self.N_pattern, self.N, self.M, self.N_defocus), complex)
                    
                    for p,q in itertools.product(range(self.N_Stokes), range(7)):
                         S_est_vec[p] += self.H_dyadic_OTF[p,q]*f_vec_f[np.newaxis,q]
                    
                elif f_tensor.ndim == 3:
                    f_vec_f = fft2(f_vec, axes=(1,2))
                    S_est_vec = np.zeros((self.N_Stokes, self.N, self.M, self.N_defocus*self.N_pattern), complex)
                    
                    for p,q in itertools.product(range(self.N_Stokes), range(7)):
                         S_est_vec[p] += self.H_dyadic_2D_OTF[p,q]*f_vec_f[q,:,:,np.newaxis]


                S_diff = S_stack_f-S_est_vec



                err[i+1] = np.sum(np.abs(S_diff)**2)
                if err[i+1]>err[i] and i>0:
                    break

                AH_S_diff = np.zeros((7,)+f_tensor.shape[1:], complex)
                
                if f_tensor.ndim == 4:
                    
                    for p,q in itertools.product(range(7), range(self.N_Stokes)):
                        AH_S_diff[p] += np.sum(np.conj(self.H_dyadic_OTF[q,p])*S_diff[q],axis=0)
                        
                    if full_fitting == True:
                        grad_x_map = -np.real(np.sum((f_tensor_p-f_tensor_n)*ifftn(AH_S_diff[2:7],axes=(1,2,3)),axis=0)*(1-p_map)*p_map)
                    else:
                        grad_x_map = -np.real(np.sum((f_tensor_p[0:4]-f_tensor_n[0:4])*ifftn(AH_S_diff[2:6],axes=(1,2,3)),axis=0)*(1-p_map)*p_map)
                        
                elif f_tensor.ndim == 3:
                    
                    for p,q in itertools.product(range(7), range(self.N_Stokes)):
                        AH_S_diff[p] += np.sum(np.conj(self.H_dyadic_2D_OTF[q,p])*S_diff[q],axis=2)
                        
                    if full_fitting == True:
                        grad_x_map = -np.real(np.sum((f_tensor_p-f_tensor_n)*ifft2(AH_S_diff[2:7],axes=(1,2)),axis=0)*(1-p_map)*p_map)
                    else:
                        grad_x_map = -np.real(np.sum((f_tensor_p[0:4]-f_tensor_n[0:4])*ifft2(AH_S_diff[2:6],axes=(1,2)),axis=0)*(1-p_map)*p_map)
                
                
                
                x_map -= grad_x_map/np.max(grad_x_map)*0.3


                print('|  %d  |  %.2e  |   %.2f   |'%(i+1,err[i+1],time.time()-tic_time))

                if i != 0:
                        ax[0].cla()
                        ax[1].cla()
                if f_tensor.ndim == 4:
                    ax[0].imshow(p_map[:,:,self.N_defocus//2],origin='lower', vmin=0, vmax=1)    
                    ax[1].imshow(np.transpose(p_map[self.N//2,:,:]),origin='lower',vmin=0, vmax=1)
                elif f_tensor.ndim == 3:
                    ax[0].imshow(p_map,origin='lower', vmin=0, vmax=1)
                    
                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)
            
            retardance_ap = np.stack([retardance_ap_p, retardance_ap_n])
            azimuth       = np.stack([azimuth_p, azimuth_n])
            theta         = np.stack([theta_p, theta_n]) 
            
            return retardance_ap, azimuth, theta, p_map

        
    
    
##############   phase computing function group   ##############    
        
    
    def Phase_recon(self, S0_stack, method='Tikhonov', reg_u = 1e-6, reg_p = 1e-6, \
                    rho = 1e-5, lambda_u = 1e-3, lambda_p = 1e-3, itr = 20, verbose=True, bg_filter=True):
        
        
        S0_stack = inten_normalization(S0_stack, bg_filter=bg_filter, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        if self.use_gpu:
            
            Hu = cp.array(self.Hu)
            Hp = cp.array(self.Hp)
            
            S0_stack_f = cp.fft.fft2(S0_stack, axes=(0,1))
            
            AHA = [cp.sum(cp.abs(Hu)**2, axis=2) + reg_u, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                   cp.sum(cp.conj(Hp)*Hu, axis=2), cp.sum(cp.abs(Hp)**2, axis=2) + reg_p]
            
            b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                     cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]
            
        else:
            
            S0_stack_f = fft2(S0_stack,axes=(0,1))
            
            AHA = [np.sum(np.abs(self.Hu)**2, axis=2) + reg_u, np.sum(np.conj(self.Hu)*self.Hp, axis=2),\
                   np.sum(np.conj(self.Hp)*self.Hu, axis=2), np.sum(np.abs(self.Hp)**2, axis=2) + reg_p]
            
            b_vec = [np.sum(np.conj(self.Hu)*S0_stack_f, axis=2), \
                     np.sum(np.conj(self.Hp)*S0_stack_f, axis=2)]
        
        
        if method == 'Tikhonov':
            
            # Deconvolution with Tikhonov regularization
            
            mu_sample, phi_sample = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            
        elif method == 'TV':
            
            # ADMM deconvolution with anisotropic TV regularization
            
            mu_sample, phi_sample = Dual_variable_ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            
        
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

                mu_sample_temp, phi_sample_temp = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)



            elif method == 'TV':

                # ADMM deconvolution with anisotropic TV regularization

                mu_sample_temp, phi_sample_temp = Dual_variable_ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)


            mu_sample[:,:,i] = mu_sample_temp.copy()
            phi_sample[:,:,i] = phi_sample_temp - phi_sample_temp.mean()
            
            
        return mu_sample, phi_sample
            
        
    
    def Phase_recon_3D(self, S0_stack, absorption_ratio=0.0, method='Tikhonov', reg_re = 1e-4, reg_im = 1e-4,\
                       rho = 1e-5, lambda_re = 1e-3, lambda_im = 1e-3, itr = 20, verbose=True):
        
        
        S0_stack = inten_normalization_3D(S0_stack)
        
        
        if self.N_pattern == 1:
            
            H_eff = self.H_re + absorption_ratio*self.H_im

            if method == 'Tikhonov':

                f_real = Single_variable_Tikhonov_deconv_3D(S0_stack, H_eff, reg_re, use_gpu=self.use_gpu, gpu_id=self.gpu_id)

            elif method == 'TV':

                f_real = Single_variable_ADMM_TV_deconv_3D(S0_stack, H_eff, rho, reg_re, lambda_re, itr, verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)

            return -f_real*self.psz/4/np.pi*self.lambda_illu
        
        else:
            
            if self.use_gpu:
            
                H_re = cp.array(self.H_re)
                H_im = cp.array(self.H_im)

                S0_stack_f = cp.fft.fftn(cp.array(S0_stack).astype('float32'), axes=(-3,-2,-1))

                AHA = [cp.sum(cp.abs(H_re)**2, axis=0) + reg_re, cp.sum(cp.conj(H_re)*H_im, axis=0),\
                       cp.sum(cp.conj(H_im)*H_re, axis=0), cp.sum(cp.abs(H_im)**2, axis=0) + reg_im]

                b_vec = [cp.sum(cp.conj(H_re)*S0_stack_f, axis=0), \
                         cp.sum(cp.conj(H_im)*S0_stack_f, axis=0)]

            else:

                S0_stack_f = fftn(S0_stack,axes=(-3,-2,-1))

                AHA = [np.sum(np.abs(self.H_re)**2, axis=0) + reg_re, np.sum(np.conj(self.H_re)*self.H_im, axis=0),\
                       np.sum(np.conj(self.H_im)*self.H_re, axis=0), np.sum(np.abs(self.H_im)**2, axis=0) + reg_im]

                b_vec = [np.sum(np.conj(self.H_re)*S0_stack_f, axis=0), \
                         np.sum(np.conj(self.H_im)*S0_stack_f, axis=0)]


            if method == 'Tikhonov':

                # Deconvolution with Tikhonov regularization
                
                f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)

            elif method == 'TV':

                # ADMM deconvolution with anisotropic TV regularization

                f_real, f_imag = Dual_variable_ADMM_TV_deconv_3D(AHA, b_vec, rho, lambda_re, lambda_im, itr, verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            
            return -f_real*self.psz/4/np.pi*self.lambda_illu, f_imag*self.psz/4/np.pi*self.lambda_illu

    
    
    
    
