import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import uniform_filter
from .util import *


def intensity_mapping(img_stack):
    img_stack_out = np.zeros_like(img_stack)
    img_stack_out[0] = img_stack[0]
    img_stack_out[1] = img_stack[4]
    img_stack_out[2] = img_stack[3]
    img_stack_out[3] = img_stack[1]
    img_stack_out[4] = img_stack[2]
    
    return img_stack_out



def Jones_sample(Ein, t, sa):
    
    Eout = np.zeros_like(Ein)
    Eout[0] = Ein[0] * (t[0]*np.cos(sa)**2 + t[1]*np.sin(sa)**2) + \
              Ein[1] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa)
    Eout[1] = Ein[0] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa) + \
              Ein[1] * (t[0]*np.sin(sa)**2 + t[1]*np.cos(sa)**2)
    
    return Eout


def Jones_to_Stokes(Ein):
    
    _, N, M = Ein.shape
    
    
    Stokes = np.zeros((4, N, M))
    Stokes[0] = np.abs(Ein[0])**2 + np.abs(Ein[1])**2
    Stokes[1] = np.abs(Ein[0])**2 - np.abs(Ein[1])**2
    Stokes[2] = np.real(Ein[0].conj()*Ein[1] + Ein[0]*Ein[1].conj())
    Stokes[3] = np.real(-1j*(Ein[0].conj()*Ein[1] - Ein[0]*Ein[1].conj()))
    
    
    return Stokes


def analyzer_output(Ein, alpha, beta):
    
    Eout = Ein[0] * np.exp(-1j*beta/2) * np.cos(alpha/2) - \
           Ein[1] * 1j * np.exp(1j*beta/2) * np.sin(alpha/2)
    
    return Eout




class waveorder_microscopy:
    
    def __init__(self, img_dim, lambda_illu, ps, NA_obj, NA_illu, z_defocus, chi, cali, bg_option):
        
        '''
        
        initialize the system parameters for phase and orders microscopy
        
        Inputs:
            
        
        '''
        
        # Basic parameter 
        self.N, self.M   = img_dim
        self.lambda_illu = lambda_illu
        self.ps          = ps
        self.z_defocus   = z_defocus.copy()
        self.NA_obj      = NA_obj
        self.NA_illu     = NA_illu
        self.N_defocus   = len(z_defocus)
        self.chi         = chi
        self.cali        = cali
        self.bg_option   = bg_option
        
        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        self.Pupil_obj = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
        self.Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_obj, self.lambda_illu, self.z_defocus)
        self.gen_WOTF()
        self.analyzer_para = np.array([[np.pi/2, np.pi], \
                                       [np.pi/2-self.chi, np.pi], \
                                       [np.pi/2, np.pi-self.chi], \
                                       [np.pi/2+self.chi, np.pi], \
                                       [np.pi/2, np.pi+self.chi]]) # [alpha, beta]
        
        self.N_channel = len(self.analyzer_para)
    
    def gen_WOTF(self):

        self.Hu = np.zeros((self.N_defocus, self.N, self.M),complex)
        self.Hp = np.zeros((self.N_defocus, self.N, self.M),complex)

        for i in range(self.N_defocus):
            H1 = ifft2(fft2(self.Source * self.Hz_det[i]).conj()*fft2(self.Hz_det[i]))
            H2 = ifft2(fft2(self.Source * self.Hz_det[i])*fft2(self.Hz_det[i]).conj())
            I_norm = np.sum(self.Source * self.Hz_det[i] * self.Hz_det[i].conj())
            self.Hu[i] = (H1 + H2)/I_norm
            self.Hp[i] = 1j*(H1-H2)/I_norm
            
    def simulate_waveorder_measurements(self, t_eigen, sa_orientation):        
        
        Stokes_out = np.zeros((self.N_defocus, 4, self.N, self.M))
        I_meas = np.zeros((self.N_defocus, self.N_channel, self.N, self.M))
        
        [idx_y, idx_x] = np.where(self.Source ==1) 
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
                
                E_field_out = ifft2(fft2(E_sample) * self.Hz_det[m])
                Stokes_out[m] += Jones_to_Stokes(E_field_out)
                
                for n in range(self.N_channel):
                    I_meas[m,n] += np.abs(analyzer_output(E_field_out, self.analyzer_para[n,0], self.analyzer_para[n,1]))**2
                    
            if np.mod(i+1, 100) == 0 or i+1 == N_source:
                print('Number of sources considered (%d / %d)'%(i+1,N_source))

            
        return I_meas, Stokes_out
    
    
    def Stokes_recon(self, I_meas):
        
        A_forward = 0.5*np.array([[1,0,0,-1], \
                          [1, np.sin(self.chi), 0, -np.cos(self.chi)], \
                          [1, 0, np.sin(self.chi), -np.cos(self.chi)], \
                          [1, -np.sin(self.chi), 0, -np.cos(self.chi)], \
                          [1, 0, -np.sin(self.chi), -np.cos(self.chi)]])
        
        A_pinv = np.linalg.pinv(A_forward)
        
        dim = I_meas.ndim 
        
        if dim == 3:
            S_image_recon = (A_pinv.dot(I_meas.reshape((5, self.N*self.M)))).reshape((4, self.N, self.M))
        else:
            S_image_recon = np.zeros((self.N_defocus, 4, self.N, self.M))
            for i in range(self.N_defocus):
                S_image_recon[i] = (A_pinv.dot(I_meas[i].reshape((5, self.N*self.M)))).reshape((4, self.N, self.M))
            
            
        return S_image_recon
    
    
    def Stokes_transform(self, S_image_recon):
        
        dim = S_image_recon.ndim
        
        if dim == 3:
            S_transformed = np.zeros((5, self.N, self.M))
            S_transformed[0] = S_image_recon[0]
            S_transformed[1] = S_image_recon[1] / S_image_recon[3]
            S_transformed[2] = S_image_recon[2] / S_image_recon[3]
            S_transformed[3] = S_image_recon[3]
            S_transformed[4] = (S_image_recon[1]**2 + S_image_recon[2]**2 + S_image_recon[3]**2)**(1/2) / S_image_recon[0] # DoP
        else:
            S_transformed = np.zeros((self.N_defocus, 5, self.N, self.M))
            S_transformed[:,0] = S_image_recon[:,0]
            S_transformed[:,1] = S_image_recon[:,1] / S_image_recon[:,3]
            S_transformed[:,2] = S_image_recon[:,2] / S_image_recon[:,3]
            S_transformed[:,3] = S_image_recon[:,3]
            S_transformed[:,4] = (S_image_recon[:,1]**2 + S_image_recon[:,2]**2 + S_image_recon[:,3]**2)**(1/2) / S_image_recon[:,0] # DoP
            
        return S_transformed
    
    
    def Polscope_bg_correction(self, S_image_tm, S_bg_tm):
        
        dim = S_image_tm.ndim
        
        if dim == 3:
            S_image_tm[0] /= S_bg_tm[0]
            S_image_tm[1] -= S_bg_tm[1]
            S_image_tm[2] -= S_bg_tm[2]
            S_image_tm[4] /= S_bg_tm[4]
        else:
            S_image_tm[:,0] /= S_bg_tm[0]
            S_image_tm[:,1] -= S_bg_tm[1]
            S_image_tm[:,2] -= S_bg_tm[2]
            S_image_tm[:,4] /= S_bg_tm[4]
        
        if self.bg_option == 'local':
            if dim == 3:
                S_image_tm[1] -= uniform_filter(S_image_tm[1], size=200)
                S_image_tm[2] -= uniform_filter(S_image_tm[2], size=200)
            else:
                for i in range(self.N_defocus):
                    S_image_tm[i,1] -= uniform_filter(S_image_tm[i,1], size=200)
                    S_image_tm[i,2] -= uniform_filter(S_image_tm[i,2], size=200)
        
        
        return S_image_tm
    
    
    
    
    def Polarization_recon(self, S_image_recon):
        
        
        dim = S_image_recon.ndim
        
        if dim == 3:
            Recon_para = np.zeros((4, self.N, self.M))
            Recon_para[0] = np.arctan2((S_image_recon[1]**2 + S_image_recon[2]**2)**(1/2) * S_image_recon[3], S_image_recon[3])  # retardance
            if self.cali == True:
                Recon_para[1] = 0.5*np.arctan2(-S_image_recon[1], -S_image_recon[2])%np.pi # slow-axis
            else:
                Recon_para[1] = 0.5*np.arctan2(-S_image_recon[1], S_image_recon[2])%np.pi # slow-axis
            Recon_para[2] = S_image_recon[0] # transmittance
            Recon_para[3] = S_image_recon[4] # DoP

        else:
        
            Recon_para = np.zeros((self.N_defocus, 4, self.N, self.M))
            for i in range(self.N_defocus):
                Recon_para[i,0] = np.arctan2((S_image_recon[i,1]**2 + S_image_recon[i,2]**2)**(1/2) * S_image_recon[i,3], S_image_recon[i,3])  # retardance
                if self.cali == True:
                    Recon_para[i,1] = 0.5*np.arctan2(-S_image_recon[i,1], -S_image_recon[i,2])%np.pi # slow-axis
                else:
                    Recon_para[i,1] = 0.5*np.arctan2(-S_image_recon[i,1], S_image_recon[i,2])%np.pi # slow-axis
                Recon_para[i,2] = S_image_recon[i,0] # transmittance
                Recon_para[i,3] = S_image_recon[i,4] # DoP

        
        return Recon_para
    
    
    def inten_normalization(self, S0_stack):
        
        S0_norm_stack = np.zeros_like(S0_stack)
        
        for i in range(self.N_defocus):
            S0_norm_stack[i] = S0_stack[i]/uniform_filter(S0_stack[i], size=self.N//2)
            S0_norm_stack[i] /= S0_norm_stack[i].mean()
            S0_norm_stack[i] -= 1
            
        return S0_norm_stack
        
    
    def Phase_recon(self, S0_stack, method='Tikhonov', reg_u = 1e-3, reg_p = 1e-3, rho = 1e-5, lambda_u = 1e-3, lambda_p = 1e-3, itr = 20, verbose=True):
        
        S0_stack = self.inten_normalization(S0_stack)
            
        
        if method == 'Tikhonov':
            
            # Deconvolution with Tikhonov regularization

            AHA = [np.sum(np.abs(self.Hu)**2, axis=0) + reg_u, np.sum(np.conj(self.Hu)*self.Hp, axis=0),\
                   np.sum(np.conj(self.Hp)*self.Hu, axis=0), np.sum(np.abs(self.Hp)**2, axis=0) + reg_p]

            S0_stack_f = fft2(S0_stack)
            b_vec = [np.sum(np.conj(self.Hu)*S0_stack_f, axis=0), \
                     np.sum(np.conj(self.Hp)*S0_stack_f, axis=0)]

            determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]

            mu_sample = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
            phi_sample = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))
            
        elif method == 'TV':
            
            # ADMM deconvolution with anisotropic TV regularization
            
            Dx = np.zeros((self.N, self.M))
            Dy = np.zeros((self.N, self.M))
            Dx[0,0] = 1; Dx[0,-1] = -1; Dx = fft2(Dx);
            Dy[0,0] = 1; Dy[-1,0] = -1; Dy = fft2(Dy);
            
            
            rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy)
            AHA = [np.sum(np.abs(self.Hu)**2, axis=0) + rho_term, np.sum(np.conj(self.Hu)*self.Hp, axis=0),\
                   np.sum(np.conj(self.Hp)*self.Hu, axis=0), np.sum(np.abs(self.Hp)**2, axis=0) + rho_term]
            
            determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]
            
            S0_stack_f = fft2(S0_stack)
            
            b_vec = [np.sum(np.conj(self.Hu)*S0_stack_f, axis=0), \
                     np.sum(np.conj(self.Hp)*S0_stack_f, axis=0)]
            
            z_para = np.zeros((4, self.N, self.M))
            u_para = np.zeros((4, self.N, self.M))
            D_vec = np.zeros((4, self.N, self.M))
            
            
            
            
            for i in range(itr):
                v_para = fft2(z_para - u_para)
                b_vec_new = [b_vec[0] + np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1],\
                             b_vec[1] + np.conj(Dx)*v_para[2] + np.conj(Dy)*v_para[3]]
                
                
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
            
        
        phi_sample -= phi_sample.mean()
        
        return mu_sample, phi_sample
    
    
    def Birefringence_recon(self, S1_stack, S2_stack, reg = 1e-3):
        
        # Birefringence deconvolution with slowly varying transmission approximation
        
        AHA = [np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=0) + reg, \
               np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=0), \
               -np.sum(self.Hu*np.conj(self.Hp) - np.conj(self.Hu)*self.Hp, axis=0), \
               np.sum(np.abs(self.Hu)**2 + np.abs(self.Hp)**2, axis=0) + reg]

        S1_stack_f = fft2(S1_stack)
        S2_stack_f = fft2(S2_stack)

        b_vec = [np.sum(-np.conj(self.Hu)*S1_stack_f + np.conj(self.Hp)*S2_stack_f, axis=0), \
                 np.sum(np.conj(self.Hp)*S1_stack_f + np.conj(self.Hu)*S2_stack_f, axis=0)]

        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]

        del_phi_s = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
        del_phi_c = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))
        
        
        Retardance = 2*(del_phi_s**2 + del_phi_c**2)**(1/2)
        
        if self.cali == True:
            slowaxis = 0.5*np.arctan2(del_phi_s, -del_phi_c)%np.pi
        else:
            slowaxis = 0.5*np.arctan2(del_phi_s, del_phi_c)%np.pi
        
        
        return Retardance, slowaxis
    
