import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import uniform_filter
from .util import *



class DPC_microscopy_2D:
    
    
    def __init__(self, img_dim, lambda_illu, ps, NA_obj, rotation_angle):
        
        '''
        
        initialize the system parameters for DPC microscopy
            
        
        '''
        
        # Basic parameter 
        self.N, self.M      = img_dim
        self.lambda_illu    = lambda_illu
        self.ps             = ps
        self.NA_obj         = NA_obj
        self.rotation_angle = rotation_angle
        self.N_source       = len(self.rotation_angle)
        
        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        self.Pupil_obj = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.gen_DPC_source()
        self.gen_WOTF()
        
        
    def gen_DPC_source(self):

        self.Source = np.zeros((self.N_source, self.N, self.M))

        for i in range(self.N_source):
            deg = self.rotation_angle[i]
            Source_temp = np.zeros((self.N,self.M))
            Source_temp[self.fyy * np.cos(np.deg2rad(deg)) - self.fxx*np.sin(np.deg2rad(deg)) > 1e-10] = 1
            self.Source[i] = Source_temp * self.Pupil_obj


    def gen_WOTF(self):

        self.Hu = np.zeros((self.N_source, self.N, self.M), complex)
        self.Hp = np.zeros((self.N_source, self.N, self.M), complex)

        for i in range(self.N_source):
            H1 = ifft2(fft2(self.Source[i] * self.Pupil_obj).conj()*fft2(self.Pupil_obj))
            H2 = ifft2(fft2(self.Source[i] * self.Pupil_obj)*fft2(self.Pupil_obj).conj())
            I_norm = np.sum(self.Source[i] * self.Pupil_obj * self.Pupil_obj.conj())
            self.Hu[i] = (H1 + H2)/I_norm
            self.Hp[i] = 1j*(H1-H2)/I_norm
        
    
    def simulate_2D_DPC_measurements(self, t_obj):
        
        I_meas = np.zeros((self.N_source, self.N, self.M))


        for i in range(self.N_source):
            [idx_y, idx_x] = np.where(self.Source[i] ==1) 
            N_pt_source = len(idx_y)
            for j in range(N_pt_source):
                plane_wave = np.exp(1j*2*np.pi*(self.fyy[idx_y[j], idx_x[j]] * self.yy +\
                                                self.fxx[idx_y[j], idx_x[j]] * self.xx))
                I_meas[i] += np.abs(ifft2(fft2(plane_wave * t_obj) * self.Pupil_obj))**2

                if np.mod(j+1, 500) == 0 or j+1 == N_pt_source:
                    print('Number of point sources considered (%d / %d) in illumination (%d / %d)'%(j+1, N_pt_source, i+1, self.N_source))
                    
        
        return I_meas
    
    
    def inten_normalization(self, I_meas):
        
        I_norm_stack = np.zeros_like(I_meas)
        
        for i in range(self.N_source):
            I_norm_stack[i] = I_meas[i]/I_meas[i].mean()
            I_norm_stack[i] -= 1
            
        return I_norm_stack
    
    def Phase_recon(self, I_meas, method='Tikhonov', reg_u = 1e-3, reg_p = 1e-3):
        
        I_norm_stack = self.inten_normalization(I_meas)
        
        AHA = [np.sum(np.abs(self.Hu)**2, axis=0) + reg_u, np.sum(np.conj(self.Hu)*self.Hp, axis=0),\
               np.sum(np.conj(self.Hp)*self.Hu, axis=0), np.sum(np.abs(self.Hp)**2, axis=0) + reg_p]

        I_norm_stack_f = fft2(I_norm_stack)
        b_vec = [np.sum(np.conj(self.Hu)*I_norm_stack_f, axis=0), \
                 np.sum(np.conj(self.Hp)*I_norm_stack_f, axis=0)]

        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]

        mu_sample = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
        phi_sample = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))
        
        
        return mu_sample, phi_sample

    
class DPC_microscopy_3D:
    
    def __init__(self, img_dim, lambda_illu, ps, psz, NA_obj, rotation_angle):
        
        '''
        
        initialize the system parameters for DPC microscopy
            
        
        '''
        
        # Basic parameter 
        self.N, self.M, self.L      = img_dim
        self.lambda_illu            = lambda_illu
        self.ps                     = ps
        self.psz                    = psz
        self.NA_obj                 = NA_obj
        self.rotation_angle         = rotation_angle
        self.N_source               = len(self.rotation_angle)
        
        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        self.Pupil_obj = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.Pupil_support = self.Pupil_obj.copy()
        self.gen_DPC_source()
        self.gen_WOTF()
    
    def gen_DPC_source(self):

        self.Source = np.zeros((self.N_source, self.N, self.M))

        for i in range(self.N_source):
            deg = self.rotation_angle[i]
            Source_temp = np.zeros((self.N,self.M))
            Source_temp[self.fyy * np.cos(np.deg2rad(deg)) - self.fxx*np.sin(np.deg2rad(deg)) > 1e-10] = 1
            self.Source[i] = Source_temp * self.Pupil_support
            
    
    