import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift


def Jones_sample(Ein, t, sa):
    
    '''
    Compute output electric field after the interaction between the sample and incident electric field.
    
    Input:
        Ein  : Incident electric field with size of (2, Ny, Nx)
        t    : eigen-transmission of the sample at its optical axis with size of (2, Ny, Nx)
        sa   : slow-axis orientation in radian with size of (Ny, Nx)
        
    Output:
        Eout : Output electric field with size od (2, Ny, Nx)
    '''
    
    Eout = np.zeros_like(Ein)
    Eout[0] = Ein[0] * (t[0]*np.cos(sa)**2 + t[1]*np.sin(sa)**2) + \
              Ein[1] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa)
    Eout[1] = Ein[0] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa) + \
              Ein[1] * (t[0]*np.sin(sa)**2 + t[1]*np.cos(sa)**2)
    
    return Eout


def Jones_to_Stokes(Ein):
    
    '''
    
    Given a coherent electric field, compute the corresponding Stokes vector.
    
    Input: 
        Ein    : Electric field with size of (2, Ny, Nx)
    
    Output:
        Stokes : Corresponding Stokes vector with size of (4, Ny, Nx)
    
    '''
    
    _, N, M = Ein.shape
    
    
    Stokes = np.zeros((4, N, M))
    Stokes[0] = np.abs(Ein[0])**2 + np.abs(Ein[1])**2
    Stokes[1] = np.abs(Ein[0])**2 - np.abs(Ein[1])**2
    Stokes[2] = np.real(Ein[0].conj()*Ein[1] + Ein[0]*Ein[1].conj())
    Stokes[3] = np.real(-1j*(Ein[0].conj()*Ein[1] - Ein[0]*Ein[1].conj()))
    
    
    return Stokes


def analyzer_output(Ein, alpha, beta):
    
    '''
    
    Compute output electric field after passing through an universal analyzer.
    
    Input: 
        Ein   : Incident electric field with size of (2, Ny, Nx)
        alpha : retardance of the first LC
        beta  : retardance of the second LC
        
    Output: 
        Eout  : Output electric field with size od (2, Ny, Nx)
    '''
    
    Eout = Ein[0] * np.exp(-1j*beta/2) * np.cos(alpha/2) - \
           Ein[1] * 1j * np.exp(1j*beta/2) * np.sin(alpha/2)
    
    return Eout



def gen_Pupil(fxx, fyy, NA, lambda_in):
    
    '''
    
    Compute pupil function given spatial frequency, NA, wavelength.
    
    Input: 
        fxx       : 2D spatial frequency array in x-dimension
        fyy       : 2D spatial frequency array in y-dimension
        NA        : numerical aperture of the pupil function
        lambda_in : wavelength of the light
        
    Output: 
        Pupil     : pupil function with the specified parameters
    
    '''
    
    N, M = fxx.shape
    
    Pupil = np.zeros((N,M))
    fr = (fxx**2 + fyy**2)**(1/2)
    Pupil[ fr < NA/lambda_in] = 1
    
    return Pupil


def gen_Hz_stack(fxx, fyy, Pupil_support, lambda_in, z_stack):
    
    '''
    
    Generate propagation kernel
    
    Input: 
        fxx           : 2D spatial frequency array in x-dimension
        fyy           : 2D spatial frequency array in y-dimension
        Pupil_support : the array that defines the support of the pupil function
        lambda_in     : wavelength of the light
        z_stack       : a list of defocused distance
        
    Output:
        Hz_stack      : corresponding propagation kernel with size of (Ny, Nx, Nz)
    
    '''
    
    N, M = fxx.shape
    N_stack = len(z_stack)
    N_defocus = len(z_stack)
    
    fr = (fxx**2 + fyy**2)**(1/2)
    
    oblique_factor = ((1 - lambda_in**2 * fr**2) *Pupil_support)**(1/2) / lambda_in
    
    Hz_stack = Pupil_support[:,:,np.newaxis] * np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:]*\
                                                      oblique_factor[:,:,np.newaxis])
    
    return Hz_stack


def gen_Greens_function_z(fxx, fyy, Pupil_support, lambda_in, z_stack):
        
    '''
    
    Generate Green's function
    
    Input: 
        fxx           : 2D spatial frequency array in x-dimension
        fyy           : 2D spatial frequency array in y-dimension
        Pupil_support : the array that defines the support of the pupil function
        lambda_in     : wavelength of the light
        z_stack       : a list of defocused distance
        
    Output:
        G_fun_z       : corresponding Green's function with size of (Ny, Nx, Nz)
    
    '''
    
    N, M = fxx.shape
    N_stack = len(z_stack)
    N_defocus = len(z_stack)
    
    fr = (fxx**2 + fyy**2)**(1/2)
    
    oblique_factor = ((1 - lambda_in**2 * fr**2) *Pupil_support)**(1/2) / lambda_in
    G_fun_z = -1j/4/np.pi* Pupil_support[:,:,np.newaxis] * \
              np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:] * \
                     oblique_factor[:,:,np.newaxis]) /(oblique_factor[:,:,np.newaxis]+1e-15)
    
    return G_fun_z


def WOTF_2D_compute(Source, Pupil, use_gpu=False, gpu_id=0):
    
    '''
    
    Compute 2D weak object transfer function (2D WOTF)
    
    Input:
        Source : Source pattern with size of (Ny, Nx)
        Pupil  : Pupil function with size of (Ny, Nx)
    
    Output:
        Hu     : absorption transfer function with size of (Ny, Nx) 
        Hp     : phase transfer function with size of (Ny, Nx)
    
    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source = cp.array(Source)
        Pupil  = cp.array(Pupil)
        
        H1 = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source * Pupil))*cp.fft.fft2(Pupil))
        H2 = cp.fft.ifft2(cp.fft.fft2(Source * Pupil)*cp.conj(cp.fft.fft2(Pupil)))
        I_norm = cp.sum(Source * Pupil * cp.conj(Pupil))
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
        
        Hu = cp.asnumpy(Hu)
        Hp = cp.asnumpy(Hp)
        
    else:
    
        H1 = ifft2(fft2(Source * Pupil).conj()*fft2(Pupil))
        H2 = ifft2(fft2(Source * Pupil)*fft2(Pupil).conj())
        I_norm = np.sum(Source * Pupil * Pupil.conj())
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
    
    return Hu, Hp

def WOTF_semi_2D_compute(Source, Pupil, Hz_det, G_fun_z, use_gpu=False, gpu_id=0):
    
    '''
    
    Compute semi-2D weak object transfer function (semi-2D WOTF)
    
    Input:
        Source  : Source pattern with size of (Ny, Nx)
        Pupil   : Pupil function with size of (Ny, Nx)
        Hz_det  : One slice of propagation kernel with size of (Ny, Nx)
        G_fun_z : One slice of scaled 2D Fourier transform of Green's function in xy-dimension with size of (Ny, Nx)
    
    Output:
        Hu     : absorption transfer function with size of (Ny, Nx) 
        Hp     : phase transfer function with size of (Ny, Nx)
    
    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source = cp.array(Source)
        Pupil  = cp.array(Pupil)        
        Hz_det = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)
        
        H1 = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source * Pupil * Hz_det))*cp.fft.fft2(Pupil * G_fun_z))
        H2 = cp.fft.ifft2(cp.fft.fft2(Source * Pupil * Hz_det)*cp.conj(cp.fft.fft2(Pupil * G_fun_z)))
        I_norm = cp.sum(Source * Pupil * cp.conj(Pupil))
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
        
        Hu = cp.asnumpy(Hu)
        Hp = cp.asnumpy(Hp)
        
    else:
    
        H1 = ifft2(fft2(Source * Pupil * Hz_det).conj()*fft2(Pupil * G_fun_z))
        H2 = ifft2(fft2(Source * Pupil * Hz_det)*fft2(Pupil * G_fun_z).conj())
        I_norm = np.sum(Source * Pupil * Pupil.conj())
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
    
    return Hu, Hp


def WOTF_3D_compute(Source, Pupil, Hz_det, G_fun_z, psz):
    
    
    '''
    
    Compute 3D weak object transfer function (2D WOTF)
    
    Input:
        Source  : Source pattern with size of (Ny, Nx)
        Pupil   : Pupil function with size of (Ny, Nx)
        Hz_det  : Propagation kernel with size of (Ny, Nx, Nz)
        G_fun_z : 2D Fourier transform of Green's function in xy-dimension with size of (Ny, Nx, Nz)
        psz     : pixel size in the z-dimension
        
    Output:
        H_re    : transfer function of real refractive index with size of (Ny, Nx, Nz) 
        H_im    : transfer function of imaginary refractive index with size of (Ny, Nx, Nz)
    
    '''
    
    
    _,_,Nz = Hz_det.shape
    
    window = ifftshift(np.hanning(Nz))

    H1 = ifft2(fft2(Source[:,:,np.newaxis] * Pupil[:,:,np.newaxis] * Hz_det, axes=(0,1)).conj()*\
               fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)), axes=(0,1))
    H1 = H1*window[np.newaxis,np.newaxis,:]
    H1 = fft(H1, axis=2)*psz
    H2 = ifft2(fft2(Source[:,:,np.newaxis] * Pupil[:,:,np.newaxis] * Hz_det, axes=(0,1))*\
               fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)).conj(), axes=(0,1))
    H2 = H2*window[np.newaxis,np.newaxis,:]
    H2 = fft(H2, axis=2)*psz

    I_norm = np.sum(Source * Pupil * Pupil.conj())
    H_re = (H1 + H2)/I_norm
    H_im = 1j*(H1-H2)/I_norm
    
    
    return H_re, H_im

