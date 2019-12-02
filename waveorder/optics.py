import numpy as np
import matplotlib.pyplot as plt
import gc
import itertools
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


def Jones_to_Stokes(Ein, use_gpu=False, gpu_id=0):
    
    '''
    
    Given a coherent electric field, compute the corresponding Stokes vector.
    
    Input: 
        Ein    : Electric field with size of (2, Ny, Nx, ...) 
    
    Output:
        Stokes : Corresponding Stokes vector with size of (4, Ny, Nx, ...)
    
    '''
    
    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        S0 = (cp.abs(Ein[0])**2 + cp.abs(Ein[1])**2)[cp.newaxis,:,:,:]
        S1 = (cp.abs(Ein[0])**2 - cp.abs(Ein[1])**2)[cp.newaxis,:,:,:]
        S2 = (cp.real(Ein[0].conj()*Ein[1] + Ein[0]*Ein[1].conj()))[cp.newaxis,:,:,:]
        S3 = (cp.real(-1j*(Ein[0].conj()*Ein[1] - Ein[0]*Ein[1].conj())))[cp.newaxis,:,:,:]
        Stokes = cp.concatenate((S0,S1,S2,S3), axis=0)
        
    else:
        Stokes = []
        Stokes.append(np.abs(Ein[0])**2 + np.abs(Ein[1])**2)                        # S0
        Stokes.append(np.abs(Ein[0])**2 - np.abs(Ein[1])**2)                        # S1
        Stokes.append(np.real(Ein[0].conj()*Ein[1] + Ein[0]*Ein[1].conj()))         # S2
        Stokes.append(np.real(-1j*(Ein[0].conj()*Ein[1] - Ein[0]*Ein[1].conj())))   # S3
        Stokes = np.array(Stokes)
    
    
    
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

def Source_subsample(Source_cont, NAx_coord, NAy_coord, subsampled_NA = 0.1):
    
    N,M = Source_cont.shape
    
    [idx_y, idx_x] = np.where(Source_cont==1)

    NAx_list = NAx_coord[idx_y, idx_x]
    NAy_list = NAy_coord[idx_y, idx_x]
    NA_list = ((NAx_list)**2 + (NAy_list)**2)**(0.5)
    NA_idx = np.argsort(NA_list)

    illu_list =[]
    
    first_idx = True
    
    for i in NA_idx:
        
        if first_idx:
            illu_list.append(i)
            first_idx = False
        elif np.product((NAx_list[i]-NAx_list[illu_list])**2 + (NAy_list[i]-NAy_list[illu_list])**2 >= subsampled_NA**2)==1:
            illu_list.append(i)


    Source_discrete = np.zeros((N,M))
    Source_discrete[idx_y[illu_list], idx_x[illu_list]] = 1
    
    
    return Source_discrete


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

def gen_dyadic_Greens_tensor_z(fxx, fyy, G_fun_z, Pupil_support, lambda_in):
    
    '''
    
    Generate forward dyadic Green's function in u_x, u_y, z space
    
    Input:
        fxx           : 2D spatial frequency array in x-dimension
        fyy           : 2D spatial frequency array in y-dimension
        G_fun_z       : forward Green's function in u_x, u_y, z space (3D)
        Pupil_support : the array that defines the support of the pupil function
        lambda_in     : wavelength of the light
    
    Output:
        G_tensor_z    : forward dyadic Green's function in u_x, u_y, z space
    '''
    
    N, M = fxx.shape
    fr = (fxx**2 + fyy**2)**(1/2)
    oblique_factor = ((1 - lambda_in**2 * fr**2) *Pupil_support)**(1/2) / lambda_in
    
    diff_filter = np.zeros((3,)+G_fun_z.shape, complex)
    diff_filter[0] = (1j*2*np.pi*fxx*Pupil_support)[...,np.newaxis]
    diff_filter[1] = (1j*2*np.pi*fyy*Pupil_support)[...,np.newaxis]
    diff_filter[2] = (1j*2*np.pi*oblique_factor)[...,np.newaxis]
    
    G_tensor_z = np.zeros((3, 3)+G_fun_z.shape, complex)
    
        
    for i in range(3):
        for j in range(3):
            G_tensor_z[i,j] = G_fun_z*diff_filter[i]*diff_filter[j]/(2*np.pi/lambda_in)**2
            if i == j:
                G_tensor_z[i,i] += G_fun_z
    
    
        
    return G_tensor_z
    
    

def gen_Greens_function_real(img_size, ps, psz, lambda_in):
    
    '''
    
    Generate Green's function in real space
    
    Input: 
        img_size  : tuple, image dimension (Ny, Nx, Nz)
        ps        : float, transverse pixel size
        psz       : float, axial pixel size
        lambda_in : float, wavelength of the light
        
    Output:
        G_real    : numpy.ndarray, corresponding real-space Green's function with size of (Ny, Nx, Nz)
    
    '''
    
    N, M, L = img_size
    
    x_r = (np.r_[:M]-M//2)*ps
    y_r = (np.r_[:N]-N//2)*ps
    z_r = (np.r_[:L]-L//2)*psz

    xx_r, yy_r, zz_r = np.meshgrid(x_r,y_r,z_r)
    
    # radial coordinate
    rho = (xx_r**2 + yy_r**2 + zz_r**2)**(0.5)

    
    # average radius of integration around r=0
    epsilon = (ps*ps*psz/np.pi/4*3)**(1/3)
    
    # wavenumber
    k = 2*np.pi/lambda_in
    
    # average value for Green's function at r=0
    V_epsilon=1/1j/k*(epsilon*np.exp(1j*k*epsilon) - 1/1j/k*(np.exp(1j*k*epsilon)-1))/ps/ps/psz
    
    
    G_real = np.exp(1j*k*rho)/(rho+1e-7)/4/np.pi
    G_real[rho==0] = V_epsilon
    
    return G_real

def gen_dyadic_Greens_tensor(G_real, ps, psz, lambda_in, space='real'):
    
    '''
    
    Generate dyadic Green's function tensor in real space or in Fourier space
    
    Input: 
        G_real    : numpy.ndarray, real space Greens function for wave equation with size of (Ny, Nx, Nz)
        ps        : float, transverse pixel size
        psz       : float, axial pixel size
        lambda_in : float, wavelength of the light
        space     : str, 'real' or 'Fourier' indicate real or Fourier space representation
        
    Output:
        G_tensor  : numpy.ndarray, corresponding real or Fourier space Green's tensor with size of (3, 3, Ny, Nx, Nz)
    
    '''
    
    N, M, L = G_real.shape
    
    fx_r = ifftshift((np.r_[:M]-M//2)/ps/M)
    fy_r = ifftshift((np.r_[:N]-N//2)/ps/N)
    fz_r = ifftshift((np.r_[:L]-L//2)/psz/L)
    fxx_r, fyy_r, fzz_r = np.meshgrid(fx_r,fy_r,fz_r)
    
    diff_filter = np.zeros((3, N, M, L), complex)
    diff_filter[0] = 1j*2*np.pi*fxx_r
    diff_filter[1] = 1j*2*np.pi*fyy_r
    diff_filter[2] = 1j*2*np.pi*fzz_r
    
    
    
    G_tensor = np.zeros((3, 3, N, M, L), complex)
    G_real_f = fftn(ifftshift(G_real))*(ps*ps*psz)
    
        
    for i in range(3):
        for j in range(3):
            G_tensor[i,j] = G_real_f*diff_filter[i]*diff_filter[j]/(2*np.pi/lambda_in)**2
            if i == j:
                G_tensor[i,i] += G_real_f
    
    
    if space == 'Fourier':
        
        return G_tensor
    
    elif space == 'real':
        
        return fftshift(ifftn(G_tensor, axes=(2,3,4)), axes=(2,3,4))/ps/ps/psz





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

def WOTF_semi_2D_compute(Source_support, Source, Pupil, Hz_det, G_fun_z, use_gpu=False, gpu_id=0):
    
    '''
    
    Compute semi-2D weak object transfer function (semi-2D WOTF)
    
    Input:
        Source_support : Source pattern support with size of (Ny, Nx)
        Source         : Source with spatial frequency modulation with size of (Ny, Nx)
        Pupil          : Pupil function with size of (Ny, Nx)
        Hz_det         : One slice of propagation kernel with size of (Ny, Nx)
        G_fun_z        : One slice of scaled 2D Fourier transform of Green's function in xy-dimension with size of (Ny, Nx)
    
    Output:
        Hu             : absorption transfer function with size of (Ny, Nx) 
        Hp             : phase transfer function with size of (Ny, Nx)
    
    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source = cp.array(Source)
        Source_support = cp.array(Source_support)
        Pupil  = cp.array(Pupil)        
        Hz_det = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)
        
        H1 = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source * Pupil * Hz_det))*cp.fft.fft2(Pupil * G_fun_z))
        H2 = cp.fft.ifft2(cp.fft.fft2(Source * Pupil * Hz_det)*cp.conj(cp.fft.fft2(Pupil * G_fun_z)))
        I_norm = cp.sum(Source_support * Pupil * cp.conj(Pupil))
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
        
        Hu = cp.asnumpy(Hu)
        Hp = cp.asnumpy(Hp)
        
    else:
    
        H1 = ifft2(fft2(Source * Pupil * Hz_det).conj()*fft2(Pupil * G_fun_z))
        H2 = ifft2(fft2(Source * Pupil * Hz_det)*fft2(Pupil * G_fun_z).conj())
        I_norm = np.sum(Source_support * Pupil * Pupil.conj())
        Hu = (H1 + H2)/I_norm
        Hp = 1j*(H1-H2)/I_norm
    
    return Hu, Hp


def WOTF_3D_compute(Source_support, Source, Pupil, Hz_det, G_fun_z, psz, use_gpu=False, gpu_id=0):
    
    
    '''
    
    Compute 3D weak object transfer function (2D WOTF)
    
    Input:
        Source_support : Source pattern support with size of (Ny, Nx)
        Source         : Source with spatial frequency modulation with size of (Ny, Nx)
        Pupil          : Pupil function with size of (Ny, Nx)
        Hz_det         : Propagation kernel with size of (Ny, Nx, Nz)
        G_fun_z        : 2D Fourier transform of Green's function in xy-dimension with size of (Ny, Nx, Nz)
        psz            : pixel size in the z-dimension
        
    Output:
        H_re           : transfer function of real refractive index with size of (Ny, Nx, Nz) 
        H_im           : transfer function of imaginary refractive index with size of (Ny, Nx, Nz)
    
    '''
    
    
    _,_,Nz = Hz_det.shape
    
    window = ifftshift(np.hanning(Nz)).astype('float32')
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source = cp.array(Source)
        Source_support = cp.array(Source_support)
        Pupil  = cp.array(Pupil)
        Hz_det = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)
        window = cp.array(window)
        
        H1 = cp.fft.ifft2(cp.conj(cp.fft.fft2((Source * Pupil)[:,:,cp.newaxis] * Hz_det, axes=(0,1)))*\
                   cp.fft.fft2(Pupil[:,:,cp.newaxis] * G_fun_z, axes=(0,1)), axes=(0,1))
        H1 = H1*window[cp.newaxis,cp.newaxis,:]
        H1 = cp.fft.fft(H1, axis=2)*psz
        H2 = cp.fft.ifft2(cp.fft.fft2((Source * Pupil)[:,:,cp.newaxis] * Hz_det, axes=(0,1))*\
                   cp.conj(cp.fft.fft2(Pupil[:,:,cp.newaxis] * G_fun_z, axes=(0,1))), axes=(0,1))
        H2 = H2*window[cp.newaxis,cp.newaxis,:]
        H2 = cp.fft.fft(H2, axis=2)*psz
    

        I_norm = cp.sum(Source_support * Pupil * cp.conj(Pupil))
        H_re = (H1 + H2)/I_norm
        H_im = 1j*(H1-H2)/I_norm
        
        H_re = cp.asnumpy(H_re)
        H_im = cp.asnumpy(H_im)
        
    else:
    
        

        H1 = ifft2(fft2((Source * Pupil)[:,:,np.newaxis] * Hz_det, axes=(0,1)).conj()*\
                   fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)), axes=(0,1))
        H1 = H1*window[np.newaxis,np.newaxis,:]
        H1 = fft(H1, axis=2)*psz
        H2 = ifft2(fft2((Source * Pupil)[:,:,np.newaxis] * Hz_det, axes=(0,1))*\
                   fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)).conj(), axes=(0,1))
        H2 = H2*window[np.newaxis,np.newaxis,:]
        H2 = fft(H2, axis=2)*psz

        I_norm = np.sum(Source_support * Pupil * Pupil.conj())
        H_re = (H1 + H2)/I_norm
        H_im = 1j*(H1-H2)/I_norm
    
    
    return H_re, H_im



def gen_geometric_inc_matrix(incident_theta, incident_phi, Source):
    
    '''
    
    Compute forward and backward matrix mapping from inclination coefficients to retardance
    
    Input:
        incident_theta           : theta spherical coordinate map in 2D spatial frequency grid with size of (Ny, Nx)
        incident_phi             : phi spherical coordinate map in 2D spatial frequency grid with size of (Ny, Nx)
        Source                   : illumination Source pattern in 2D spatial frequency grid with size of (N_pattern, Ny, Nx)
    
    Output: 
        geometric_inc_matrix     : forward matrix mapping from inclination coefficients to retardance
        geometric_inc_matrix_inv : pinv of the forward matrix
    
    '''
    
    N_pattern,_,_ = Source.shape
    
    geometric_inc_matrix = []

    for i in range(N_pattern):

        idx_y, idx_x = np.where(Source[i])

        geometric_inc_matrix.append([1, np.mean(0.5*np.cos(2*incident_theta[idx_y,idx_x])),\
                                   np.mean(-0.5*np.sin(2*incident_theta[idx_y,idx_x])*np.cos(incident_phi[idx_y,idx_x])), \
                                   np.mean(-0.5*np.sin(2*incident_theta[idx_y,idx_x])*np.sin(incident_phi[idx_y,idx_x])), \
                                   np.mean(-0.5*(np.sin(incident_theta[idx_y,idx_x])**2)*np.cos(2*incident_phi[idx_y,idx_x])),\
                                   np.mean(-0.5*(np.sin(incident_theta[idx_y,idx_x])**2)*np.sin(2*incident_phi[idx_y,idx_x]))])


    geometric_inc_matrix = np.array(geometric_inc_matrix)
    geometric_inc_matrix_inv = np.linalg.pinv(geometric_inc_matrix)
    
    
    
    return geometric_inc_matrix, geometric_inc_matrix_inv



def SEAGLE_vec_forward(E_tot, f_scat_tensor, G_tensor, use_gpu=False, gpu_id=0):
    
    '''
    
    Compute vectorial SEAGLE forward model
    
    Input:
        E_tot         : ndarray, total electric field (scattered + incident) with size of (3, Ny, Nx, Nz)
        f_scat_tensor : ndarray, scattering potential tensor with size of (3, 3, Ny, Nx, Nz)
        G_tensor      : ndarray, dyadic Green's function with size of (3, 3, Ny, Nx, Nz)
        use_gpu       : bool, option to use gpu or not
        gpu_id        : int, gpu_id for computation
        
    Output:
        E_in_est      : ndarray, estimated incident electric field with size of (3, Ny, Nx, Nz)
        
    '''
    
    N, M, L = E_tot.shape[1:]
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        pad_convolve_G = lambda x, y, z: cp.fft.ifftn(cp.fft.fftn(cp.pad(x,((N//2,N//2),(M//2,M//2),(L//2,L//2)), \
                                                       mode='constant', constant_values=y))*z)[N//2:-N//2,M//2:-M//2,L//2:-L//2]
        E_interact = cp.zeros((3, N, M, L), complex)
        E_in_est = cp.zeros_like(E_tot, complex)


        for p, q in itertools.product(range(3), range(3)):
            E_interact[p] += f_scat_tensor[p,q]*E_tot[q]

        for p, q in itertools.product(range(3), range(3)):     
            E_in_est[p] += pad_convolve_G(E_interact[q], cp.asnumpy(cp.abs(cp.mean(E_interact[q]))), G_tensor[p,q])
            if p == q:
                E_in_est[p] +=  E_tot[p]
        
    else: 
    
        pad_convolve_G = lambda x, y, z: ifftn(fftn(np.pad(x,((N//2,),(M//2,),(L//2,)), \
                                                           mode='constant', constant_values=y))*z)[N//2:-N//2,M//2:-M//2,L//2:-L//2]
        E_interact = np.zeros((3, N, M, L), complex)
        E_in_est = np.zeros_like(E_tot, complex)


        for p, q in itertools.product(range(3), range(3)):
            E_interact[p] += f_scat_tensor[p,q]*E_tot[q]

        for p, q in itertools.product(range(3), range(3)):     
            E_in_est[p] += pad_convolve_G(E_interact[q], np.abs(np.mean(E_interact[q])), G_tensor[p,q])
            if p == q:
                E_in_est[p] +=  E_tot[p]
    
        
    return E_in_est



def SEAGLE_vec_backward(E_diff, f_scat_tensor, G_tensor, use_gpu=False, gpu_id=0):
    
    '''
    
    Compute the adjoint of vectorial SEAGLE forward model
    
    Input:
        E_diff        : ndarray, difference between estimated and real incident field with size of (3, Ny, Nx, Nz)
        f_scat_tensor : ndarray, scattering potential tensor with size of (3, 3, Ny, Nx, Nz)
        G_tensor      : ndarray, dyadic Green's function with size of (3, 3, Ny, Nx, Nz)
        use_gpu       : bool, option to use gpu or not
        gpu_id        : int, gpu_id for computation
        
    Output:
        grad_E        : ndarray, gradient of the total electric field with size of (3, Ny, Nx, Nz)
        
    '''
    
    N, M, L = E_diff.shape[1:]
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        pad_convolve_G = lambda x, y, z: cp.fft.ifftn(cp.fft.fftn(cp.pad(x,((N//2,N//2),(M//2,M//2),(L//2,L//2)), \
                                                       mode='constant', constant_values=y))*z)[N//2:-N//2,M//2:-M//2,L//2:-L//2]
        
        E_diff_conv = cp.zeros_like(E_diff, complex)
        grad_E = cp.zeros_like(E_diff, complex)


        for p, q in itertools.product(range(3), range(3)):
            E_diff_conv[p] += pad_convolve_G(E_diff[q], cp.asnumpy(cp.abs(cp.mean(E_diff[p]))), G_tensor[p,q].conj())


        for p in range(3):
            E_interact = cp.zeros((N,M,L), complex)
            for q in range(3):
                E_interact += f_scat_tensor[q,p].conj()*E_diff_conv[q]
            grad_E[p] = E_diff[p] + E_interact
        
    else:
    
    
        pad_convolve_G = lambda x, y, z: ifftn(fftn(np.pad(x,((N//2,),(M//2,),(L//2,)), \
                                                           mode='constant', constant_values=y))*z)[N//2:-N//2,M//2:-M//2,L//2:-L//2]

        E_diff_conv = np.zeros_like(E_diff, complex)
        grad_E = np.zeros_like(E_diff, complex)


        for p, q in itertools.product(range(3), range(3)):
            E_diff_conv[p] += pad_convolve_G(E_diff[q], np.abs(np.mean(E_diff[p])), G_tensor[p,q].conj())


        for p in range(3):
            E_interact = np.zeros((N,M,L), complex)
            for q in range(3):
                E_interact += f_scat_tensor[q,p].conj()*E_diff_conv[q]
            grad_E[p] = E_diff[p] + E_interact
        
    return grad_E

