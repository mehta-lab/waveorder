import numpy as np
import matplotlib.pyplot as plt
import pywt
import time

from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import uniform_filter


import re
numbers = re.compile(r'(\d+)')





def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



def genStarTarget(N, M, blur_px = 2, margin=60):
    
    '''
    
    generate Siemens star image for simulation
    
    Parameters
    ----------
        N       : int
                  size of the simulated image in y dimension
             
        M       : int
                  size of the simulated image in x dimension
        
        blur_px : float
                  the standard deviation of the imposed Gaussian blur on the simulated image
                  
        margin  : int
                  the size of blank margin on the simulated image
        
    Returns
    -------
        star    : numpy.ndarray
                  Siemens star with the size of (Ny, Nx)
                  
        theta   : numpy.ndarray
                  azimuthal angle of the polar coordinate with the size of (Ny, Nx)
                  
        xx      : numpy.ndarray
                  x coordinate array with the size of (Ny, Nx)
    
    '''

    
    # Construct Siemens star

    x = np.r_[:N]-N//2
    y = np.r_[:M]-M//2

    xx, yy = np.meshgrid(x,y)

    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # star = (1 + np.cos(40*theta))
    # star = np.pad(star[10:-10,10:-10],(10,),mode='constant')
    star = (1 + np.cos(16*theta))
    star = np.pad(star[margin:-margin,margin:-margin],(margin,),mode='constant')
    star[star<1] = 0
    
    # Filter to prevent aliasing


    Gaussian = np.exp(-rho**2/(2*blur_px**2))
    
    star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))
    # star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))*(2+np.sin(2*np.pi*(1/5)*rho))
    star /= np.max(star)
    
    
    return star, theta, xx

def genStarTarget_3D(img_dim, ps, psz, blur_size = 0.1, inc_upper_bound=np.pi/8, inc_range=np.pi/64):
    
    '''
    
    generate 3D star image for simulation
    
    Parameters
    ----------
        img_dim         : tuple
                          shape of the computed 3D space with size of (Ny, Nx, Nz)
                          
        ps              : float
                          transverse pixel size of the image space
                          
        psz             : float
                          axial step size of the image space
        
        blur_size       : float
                          the standard deviation of the imposed 3D Gaussian blur on the simulated image
                  
        inc_upper_bound : float
                          the upper bound of the inclination angle of the tilted feature from 0 to pi/2
        
        inc_range       : float
                          the range of the inclination that defines the axial width of the tilted feature
        
    Returns
    -------
        star            : numpy.ndarray
                          3D star image with the size of (Ny, Nx, Nz)
                  
        azimuth         : numpy.ndarray
                          azimuthal angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)
                  
        inc_angle       : numpy.ndarray
                          theta angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)
    
    '''
    
    
    N,M,L = img_dim
    
    x = (np.r_[:M]-M//2)*ps
    y = (np.r_[:N]-N//2)*ps
    z = (np.r_[:L]-L//2)*psz

    xx, yy, zz = np.meshgrid(x,y,z)

    rho = np.sqrt(xx**2 + yy**2 + zz**2)
    azimuth = np.arctan2(yy, xx)
    inc_angle = np.arctan2((xx**2 + yy**2)**(1/2), zz)


    star = (1 + np.cos(16*azimuth))

    star = np.pad(star[20:-20,20:-20,20:-20],((20,),(20,),(20,)),mode='constant')
    star[star<1] = 0
    star[np.abs(inc_angle-np.pi/2)>inc_upper_bound] = 0
    star[np.abs(inc_angle-np.pi/2)<inc_upper_bound-inc_range] = 0

    # Filter to prevent aliasing

    Gaussian = np.exp(-rho**2/(2*blur_size**2))

    star = np.maximum(0, np.real(ifftn(fftn(star) * fftn(ifftshift(Gaussian)))))
    star /= np.max(star)
    
    
    return star, azimuth, inc_angle


def gen_sphere_target(img_dim, ps, psz, radius, blur_size = 0.1):
    
    '''
    
    generate 3D sphere target for simulation
    
    Parameters
    ----------
        img_dim   : tuple
                    shape of the computed 3D space with size of (Ny, Nx, Nz)
                          
        ps        : float
                    transverse pixel size of the image space
                          
        psz       : float
                    axial step size of the image space
                          
        radius    : float
                    radius of the generated sphere
        
        blur_size : float
                    the standard deviation of the imposed 3D Gaussian blur on the simulated image
                  
        
    Returns
    -------
        sphere    : numpy.ndarray
                    3D star image with the size of (Ny, Nx, Nz)
                  
        azimuth   : numpy.ndarray
                    azimuthal angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)
                  
        inc_angle : numpy.ndarray
                    theta angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)
    
    '''
    
    
    N,M,L = img_dim
    x = (np.r_[:M]-M//2)*ps
    y = (np.r_[:N]-N//2)*ps
    z = (np.r_[:L]-L//2)*psz
    
    xx, yy, zz = np.meshgrid(x,y,z)
    
    rho = np.sqrt(xx**2 + yy**2 + zz**2)
    azimuth = np.arctan2(yy, xx)
    inc_angle = np.arctan2((xx**2 + yy**2)**(1/2), zz)
    
    sphere = np.zeros_like(xx)
    sphere[xx**2 + yy**2 + zz**2<radius**2] = 1
    
    Gaussian = np.exp(-rho**2/(2*blur_size**2))
    
    sphere = np.maximum(0, np.real(ifftn(fftn(sphere,axes=(0,1,2)) * \
                                         fftn(ifftshift(Gaussian),axes=(0,1,2)),axes=(0,1,2))))
    sphere /= np.max(sphere)
    
    
    return sphere, azimuth, inc_angle



def gen_coordinate(img_dim, ps):
    
    '''
    
    generate spatial and spatial frequency coordinate arrays
    
    Input:
        img_dim : tuple
                  shape of the computed 2D space with size of (Ny, Nx)
                    
        ps      : float
                  transverse pixel size of the image space
        
    Output:
        xx      : numpy.ndarray
                  x coordinate array with the size of (Ny, Nx)
                  
        yy      : numpy.ndarray
                  y coordinate array with the size of (Ny, Nx)
        
        fxx     : numpy.ndarray
                  x component of 2D spatial frequency array with the size of (Ny, Nx)
                        
        fyy     : numpy.ndarray
                  y component of 2D spatial frequency array with the size of (Ny, Nx)
    
    '''
    
    N, M = img_dim
    
    fx = ifftshift((np.r_[:M]-M/2)/M/ps)
    fy = ifftshift((np.r_[:N]-N/2)/N/ps)
    x  = ifftshift((np.r_[:M]-M/2)*ps)
    y  = ifftshift((np.r_[:N]-N/2)*ps)


    xx, yy = np.meshgrid(x, y)
    fxx, fyy = np.meshgrid(fx, fy)

    return (xx, yy, fxx, fyy)


def axial_upsampling(I_meas, upsamp_factor=1):
    
    F = lambda x: ifftshift(fft(fftshift(x,axes=2),axis=2),axes=2)
    iF = lambda x: ifftshift(ifft(fftshift(x,axes=2),axis=2),axes=2)
    
    
    N, M, Lcrop = I_meas.shape
    L = Lcrop*upsamp_factor
    
    I_meas_up = np.zeros((N,M,L))
    if (L-Lcrop)//2 == 0:
        I_meas_up = np.abs(iF(np.pad(F(I_meas), ((0,),(0,),((L-Lcrop)//2,)),mode='constant')))
    else:
        I_meas_up = np.abs(iF(np.pad(F(I_meas), ((0,0),(0,0),((L-Lcrop)//2+1,(L-Lcrop)//2)),mode='constant')))
    
    return I_meas_up

def softTreshold(x, threshold, use_gpu=False, gpu_id=0):
    
    '''
    
    compute soft thresholding operation on numpy ndarray with gpu option
    
    Parameters
    ----------
        x          : numpy.ndarray
                     targeted array for soft thresholding operation with arbitrary size
                  
        threshold  : numpy.ndarray
                     array contains threshold value for each x array position
        
        use_gpu    : bool
                     option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        x_threshold : numpy.ndarray
                      thresholded array
                      
    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        magnitude = cp.abs(x)
        ratio = cp.maximum(0, magnitude-threshold) / (magnitude+1e-16)
    else:
        magnitude = np.abs(x)
        ratio = np.maximum(0, magnitude-threshold) / (magnitude+1e-16)
        
    x_threshold = x*ratio
    
    return x_threshold


def wavelet_softThreshold(img, wavelet, threshold, level = 1):
    
    '''
    
    soft thresholding in the nD wavelet space
    
    Parameters
    ----------
        img       : numpy.ndarray
                    image or volume in nD space
                    
        wavelet   : str
                    type of wavelet to use (pywt.wavelist() to find the whole list)
                    
        threshold : float
                    threshold value 
        
    Returns
    -------
        img_thres : numpy.ndarray
                    denoised image or volume in nD space
    
    '''
    
    coeffs = pywt.wavedecn(img, wavelet, level=level)
    
    for i in range(level+1):
        if i == 0:
            coeffs[i] = softTreshold(coeffs[i], threshold)
        else:
            for item in coeffs[i]:
                coeffs[i][item] = softTreshold(coeffs[i][item], threshold)

    img_thres = pywt.waverecn(coeffs, wavelet)
    
    return img_thres


def array_based_4x4_det(a):
    
    '''
    
    compute array-based determinant on 4 x 4 matrix
    
    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              4 x 4 matrix in the nD space with the shape of (4, 4, Ny, Nx, Nz, ...)
        
    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)
    
    '''
    
    sub_det1 = a[0,0]*(a[1,1]*(a[2,2]*a[3,3]-a[3,2]*a[2,3]) - \
                       a[1,2]*(a[2,1]*a[3,3]-a[3,1]*a[2,3]) + \
                       a[1,3]*(a[2,1]*a[3,2]-a[3,1]*a[2,2]) )
    
    sub_det2 = a[0,1]*(a[1,0]*(a[2,2]*a[3,3]-a[3,2]*a[2,3]) - \
                       a[1,2]*(a[2,0]*a[3,3]-a[3,0]*a[2,3]) + \
                       a[1,3]*(a[2,0]*a[3,2]-a[3,0]*a[2,2]) )
    
    sub_det3 = a[0,2]*(a[1,0]*(a[2,1]*a[3,3]-a[3,1]*a[2,3]) - \
                       a[1,1]*(a[2,0]*a[3,3]-a[3,0]*a[2,3]) + \
                       a[1,3]*(a[2,0]*a[3,1]-a[3,0]*a[2,1]) )
    
    sub_det4 = a[0,3]*(a[1,0]*(a[2,1]*a[3,2]-a[3,1]*a[2,2]) - \
                       a[1,1]*(a[2,0]*a[3,2]-a[3,0]*a[2,2]) + \
                       a[1,2]*(a[2,0]*a[3,1]-a[3,0]*a[2,1]) )
    
    det = sub_det1 - sub_det2 + sub_det3 - sub_det4

    
    return det

def array_based_5x5_det(a):
    
    '''
    
    compute array-based determinant on 5 x 5 matrix
    
    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              5 x 5 matrix in the nD space with the shape of (5, 5, Ny, Nx, Nz, ...)
        
    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)
    
    '''
    
    det = a[0,0]*array_based_4x4_det(a[1:,1:]) - \
          a[0,1]*array_based_4x4_det(a[1:,[0,2,3,4]]) + \
          a[0,2]*array_based_4x4_det(a[1:,[0,1,3,4]]) - \
          a[0,3]*array_based_4x4_det(a[1:,[0,1,2,4]]) + \
          a[0,4]*array_based_4x4_det(a[1:,[0,1,2,3]])
    
    return det

def array_based_6x6_det(a):
    
    '''
    
    compute array-based determinant on 6 x 6 matrix
    
    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              6 x 6 matrix in the nD space with the shape of (6, 6, Ny, Nx, Nz, ...)
        
    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)
    
    '''
    
    det = a[0,0]*array_based_5x5_det(a[1:,1:]) - \
          a[0,1]*array_based_5x5_det(a[1:,[0,2,3,4,5]]) + \
          a[0,2]*array_based_5x5_det(a[1:,[0,1,3,4,5]]) - \
          a[0,3]*array_based_5x5_det(a[1:,[0,1,2,4,5]]) + \
          a[0,4]*array_based_5x5_det(a[1:,[0,1,2,3,5]]) - \
          a[0,5]*array_based_5x5_det(a[1:,[0,1,2,3,4]])
    
    return det

def array_based_7x7_det(a):
    
    '''
    
    compute array-based determinant on 7 x 7 matrix
    
    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              7 x 7 matrix in the nD space with the shape of (7, 7, Ny, Nx, Nz, ...)
        
    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)
    
    '''
    
    det = a[0,0]*array_based_6x6_det(a[1:,1:]) - \
          a[0,1]*array_based_6x6_det(a[1:,[0,2,3,4,5,6]]) + \
          a[0,2]*array_based_6x6_det(a[1:,[0,1,3,4,5,6]]) - \
          a[0,3]*array_based_6x6_det(a[1:,[0,1,2,4,5,6]]) + \
          a[0,4]*array_based_6x6_det(a[1:,[0,1,2,3,5,6]]) - \
          a[0,5]*array_based_6x6_det(a[1:,[0,1,2,3,4,6]]) + \
          a[0,6]*array_based_6x6_det(a[1:,[0,1,2,3,4,5]])
    
    return det
    


def uniform_filter_2D(image, size, use_gpu=False, gpu_id=0):
    
    '''
    
    compute uniform filter operation on 2D image with gpu option
    
    Parameters
    ----------
        image          : numpy.ndarray
                         targeted image for filtering with size of (Ny, Nx) 
                  
        size           : int
                         size of the kernel for uniform filtering
        
        use_gpu        : bool
                         option to use gpu or not
        
        gpu_id         : int
                         number refering to which gpu will be used
    
    Returns
    -------
        image_filtered : numpy.ndarray
                         filtered image with size of (Ny, Nx)
                         
    '''
    
    N, M = image.shape
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        # filter in y direction
        
        image_cp = cp.array(image)
    
        kernel_y = cp.zeros((3*N,))
        kernel_y[3*N//2-size//2:3*N//2+size//2] = 1
        kernel_y /= cp.sum(kernel_y)
        kernel_y = cp.fft.fft(cp.fft.ifftshift(kernel_y))

        image_bound_y = cp.zeros((3*N,M))
        image_bound_y[N:2*N,:] = image_cp.copy()
        image_bound_y[0:N,:] = cp.flipud(image_cp)
        image_bound_y[2*N:3*N,:] = cp.flipud(image_cp)
        filtered_y = cp.real(cp.fft.ifft(cp.fft.fft(image_bound_y,axis=0)*kernel_y[:,cp.newaxis],axis=0))
        filtered_y = filtered_y[N:2*N,:]
        
        # filter in x direction
        
        kernel_x = cp.zeros((3*M,))
        kernel_x[3*M//2-size//2:3*M//2+size//2] = 1
        kernel_x /= cp.sum(kernel_x)
        kernel_x = cp.fft.fft(cp.fft.ifftshift(kernel_x))

        image_bound_x = cp.zeros((N,3*M))
        image_bound_x[:,M:2*M] = filtered_y.copy()
        image_bound_x[:,0:M] = cp.fliplr(filtered_y)
        image_bound_x[:,2*M:3*M] = cp.fliplr(filtered_y)

        image_filtered = cp.real(cp.fft.ifft(cp.fft.fft(image_bound_x,axis=1)*kernel_x[cp.newaxis,:],axis=1))
        image_filtered = image_filtered[:,M:2*M]
    else:
        image_filtered = uniform_filter(image, size=size)
        
        
    return image_filtered


def inten_normalization(img_stack, bg_filter=True, use_gpu=False, gpu_id=0):
    
    '''
    
    layer-by-layer intensity normalization to reduce low-frequency phase artifacts
    
    Parameters
    ----------
        img_stack      : numpy.ndarray
                         image stack for normalization with size of (Ny, Nx, Nz)
                  
        type           : str
                         '2D' refers to layer-by-layer and '3D' refers to whole-stack normalization
                     
        bg_filter      : bool
                         option for slow-varying 2D background normalization with uniform filter
        
        use_gpu        : bool
                         option to use gpu or not
        
        gpu_id         : int
                         number refering to which gpu will be used
    
    Returns
    -------
        img_norm_stack : numpy.ndarray
                         normalized image stack with size of (Ny, Nx, Nz)
                         
    '''
        
    N,M, Nimg = img_stack.shape

    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        img_stack = cp.array(img_stack)
        img_norm_stack = cp.zeros_like(img_stack)

        for i in range(Nimg):
            if bg_filter:
                img_norm_stack[:,:,i] = img_stack[:,:,i]/uniform_filter_2D(img_stack[:,:,i], size=N//2, use_gpu=True, gpu_id=gpu_id)
            else:
                img_norm_stack[:,:,i] = img_stack[:,:,i].copy()
            img_norm_stack[:,:,i] /= img_norm_stack[:,:,i].mean()
            img_norm_stack[:,:,i] -= 1

    else:
        img_norm_stack = np.zeros_like(img_stack)

        for i in range(Nimg):
            if bg_filter:
                img_norm_stack[:,:,i] = img_stack[:,:,i]/uniform_filter(img_stack[:,:,i], size=N//2)
            else:
                img_norm_stack[:,:,i] = img_stack[:,:,i].copy()
            img_norm_stack[:,:,i] /= img_norm_stack[:,:,i].mean()
            img_norm_stack[:,:,i] -= 1

    return img_norm_stack



def inten_normalization_3D(img_stack):
    
    '''
    
    whole-stack intensity normalization to reduce low-frequency phase artifacts
    
    Parameters
    ----------
        img_stack      : numpy.ndarray
                         image stack for normalization with size of (Ny, Nx, Nz)
    
    Returns
    -------
        img_norm_stack : numpy.ndarray
                         normalized image stack with size of (Ny, Nx, Nz)
                         
    '''


    img_norm_stack = np.zeros_like(img_stack)
    img_norm_stack = img_stack / np.mean(img_stack,axis=(-3,-2,-1))[...,np.newaxis,np.newaxis,np.newaxis]
    img_norm_stack -= 1

    return img_norm_stack


def Dual_variable_Tikhonov_deconv_2D(AHA, b_vec, determinant=None, use_gpu=False, gpu_id=0, move_cpu=True):
    
    '''
    
    2D Tikhonov deconvolution to solve for phase and absorption with weak object transfer function
    
    Parameters
    ----------
        AHA         : list
                      A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                      | AHA[0]  AHA[1] |
                      | AHA[2]  AHA[3] |
                  
        b_vec       : list
                      measured intensity stored with a list of 2 2D numpy array (2 vectors)
                      | b_vec[0] |
                      | b_vec[1] |
                     
        determinant : numpy.ndarray
                      determinant of the AHA matrix in 2D space
        
        use_gpu     : bool
                      option to use gpu or not
        
        gpu_id      : int
                      number refering to which gpu will be used
                     
        move_cpu    : bool
                      option to move the array from gpu to cpu
    
    Returns
    -------
        mu_sample   : numpy.ndarray
                      2D absorption reconstruction with the size of (Ny, Nx)
                  
        phi_sample  : numpy.ndarray
                      2D phase reconstruction with the size of (Ny, Nx)
    '''
    
    if determinant is None:
        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]
    
    mu_sample_f = (b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant
    phi_sample_f = (b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant

    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        mu_sample = cp.real(cp.fft.ifft2(mu_sample_f))
        phi_sample = cp.real(cp.fft.ifft2(phi_sample_f))
        
        if move_cpu:
            mu_sample = cp.asnumpy(mu_sample)
            phi_sample = cp.asnumpy(phi_sample)
        
    else:
        mu_sample = np.real(ifft2(mu_sample_f))
        phi_sample = np.real(ifft2(phi_sample_f))

    return mu_sample, phi_sample




def Dual_variable_ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose, use_gpu=False, gpu_id=0):
    
    '''
    
    2D TV deconvolution to solve for phase and absorption with weak object transfer function
    
    ADMM formulation:
        
        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2
    
    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                     | AHA[0]  AHA[1] |
                     | AHA[2]  AHA[3] |
                  
        b_vec      : list
                     measured intensity stored with a list of 2 2D numpy array (2 vectors)
                     | b_vec[0] |
                     | b_vec[1] |
                     
        rho        : float
                     ADMM rho parameter
        
        lambda_u   : float
                     TV regularization parameter for absorption
        
        lambda_p   : float
                     TV regularization parameter for phase
        
        itr        : int
                     number of iterations of ADMM algorithm
        
        verbose    : bool
                     option to display progress of the computation
        
        use_gpu    : bool
                     option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        mu_sample  : numpy.ndarray
                     2D absorption reconstruction with the size of (Ny, Nx)
                  
        phi_sample : numpy.ndarray
                     2D phase reconstruction with the size of (Ny, Nx)
    '''

    # ADMM deconvolution with anisotropic TV regularization
    
    N, M = b_vec[0].shape
    Dx = np.zeros((N, M)); Dx[0,0] = 1; Dx[0,-1] = -1;
    Dy = np.zeros((N, M)); Dy[0,0] = 1; Dy[-1,0] = -1;

    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Dx = cp.fft.fft2(cp.array(Dx));
        Dy = cp.fft.fft2(cp.array(Dy));

        rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy)

        z_para = cp.zeros((4, N, M))
        u_para = cp.zeros((4, N, M))
        D_vec = cp.zeros((4, N, M))


    else:
        Dx = fft2(Dx);
        Dy = fft2(Dy);

        rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy)

        z_para = np.zeros((4, N, M))
        u_para = np.zeros((4, N, M))
        D_vec = np.zeros((4, N, M))


    AHA[0] = AHA[0] + rho_term
    AHA[3] = AHA[3] + rho_term

    determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]


    for i in range(itr):


        if use_gpu:

            v_para = cp.fft.fft2(z_para - u_para)
            b_vec_new = [b_vec[0] + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1]),\
                         b_vec[1] + rho*(cp.conj(Dx)*v_para[2] + cp.conj(Dy)*v_para[3])]

            mu_sample, phi_sample = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec_new, determinant=determinant, \
                                                                     use_gpu=use_gpu, gpu_id=gpu_id, move_cpu=not use_gpu)
            

            D_vec[0] = mu_sample - cp.roll(mu_sample, -1, axis=1)
            D_vec[1] = mu_sample - cp.roll(mu_sample, -1, axis=0)
            D_vec[2] = phi_sample - cp.roll(phi_sample, -1, axis=1)
            D_vec[3] = phi_sample - cp.roll(phi_sample, -1, axis=0)


            z_para = D_vec + u_para

            z_para[:2,:,:] = softTreshold(z_para[:2,:,:], lambda_u/rho, use_gpu=True, gpu_id=gpu_id)
            z_para[2:,:,:] = softTreshold(z_para[2:,:,:], lambda_p/rho, use_gpu=True, gpu_id=gpu_id)

            u_para += D_vec - z_para

            if i == itr-1:
                mu_sample  = cp.asnumpy(mu_sample)
                phi_sample = cp.asnumpy(phi_sample)




        else:

            v_para = fft2(z_para - u_para)
            b_vec_new = [b_vec[0] + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1]),\
                         b_vec[1] + rho*(np.conj(Dx)*v_para[2] + np.conj(Dy)*v_para[3])]
            
            mu_sample, phi_sample = Dual_variable_Tikhonov_deconv_2D(AHA, b_vec_new, determinant=determinant)
            
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



def Single_variable_Tikhonov_deconv_3D(S0_stack, H_eff, reg_re, use_gpu=False, gpu_id=0):
    
    '''
    
    3D Tikhonov deconvolution to solve for phase with weak object transfer function
    
    Parameters
    ----------
        S0_stack : numpy.ndarray
                   S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)
                  
        H_eff    : numpy.ndarray
                   effective transfer function with size of (Ny, Nx, Nz)
                     
        reg_re   : float
                   Tikhonov regularization parameter
        
        use_gpu  : bool
                   option to use gpu or not
        
        gpu_id   : int
                   number refering to which gpu will be used
    
    Returns
    -------
        f_real   : numpy.ndarray
                   3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)
    '''
    
    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        S0_stack_f = cp.fft.fftn(cp.array(S0_stack.astype('float32')), axes=(-3,-2,-1))
        H_eff      = cp.array(H_eff.astype('complex64'))
        
        f_real     = cp.asnumpy(cp.real(cp.fft.ifftn(S0_stack_f * cp.conj(H_eff) / (cp.abs(H_eff)**2 + reg_re),axes=(-3,-2,-1))))
    else:
        
        S0_stack_f = fftn(S0_stack, axes=(-3,-2,-1))
        
        f_real     = np.real(ifftn(S0_stack_f * np.conj(H_eff) / (np.abs(H_eff)**2 + reg_re),axes=(-3,-2,-1)))
        
    return f_real


def Dual_variable_Tikhonov_deconv_3D(AHA, b_vec, determinant=None, use_gpu=False, gpu_id=0, move_cpu=True):
    
    '''
    
    3D Tikhonov deconvolution to solve for phase and absorption with weak object transfer function
    
    Parameters
    ----------
        AHA         : list
                      A^H times A matrix stored with a list of 4 3D numpy array (4 diagonal matrices)
                      | AHA[0]  AHA[1] |
                      | AHA[2]  AHA[3] |
                  
        b_vec       : list
                      measured intensity stored with a list of 2 3D numpy array (2 vectors)
                      | b_vec[0] |
                      | b_vec[1] |
                     
        determinant : numpy.ndarray
                      determinant of the AHA matrix in 3D space
        
        use_gpu     : bool
                      option to use gpu or not
        
        gpu_id      : int
                      number refering to which gpu will be used
                     
        move_cpu    : bool
                      option to move the array from gpu to cpu
    
    Returns
    -------
        f_real      : numpy.ndarray
                      3D real scattering potential (unscaled phase) reconstruction with the size of (Ny, Nx, Nz)
                  
        f_imag      : numpy.ndarray
                      3D imaginary scattering potential (unscaled absorption) reconstruction with the size of (Ny, Nx, Nz)
    '''
    
    if determinant is None:
        determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]
    
    f_real_f = (b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant
    f_imag_f = (b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant

    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        f_real = cp.real(cp.fft.ifftn(f_real_f))
        f_imag = cp.real(cp.fft.ifftn(f_imag_f))
        
        if move_cpu:
            f_real = cp.asnumpy(f_real)
            f_imag = cp.asnumpy(f_imag)
        
    else:
        f_real = np.real(ifftn(f_real_f))
        f_imag = np.real(ifftn(f_imag_f))

    return f_real, f_imag



def Single_variable_ADMM_TV_deconv_3D(S0_stack, H_eff, rho, reg_re, lambda_re, itr, verbose, use_gpu=False, gpu_id=0):
    
    '''
    
    3D TV deconvolution to solve for phase with weak object transfer function
    
    ADMM formulation:
        
        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2
    
    Parameters
    ----------
        S0_stack  : numpy.ndarray
                    S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)
                  
        H_eff     : numpy.ndarray
                    effective transfer function with size of (Ny, Nx, Nz)
                     
        reg_re    : float
                    Tikhonov regularization parameter
                     
        rho       : float
                    ADMM rho parameter
        
        lambda_re : float
                    TV regularization parameter for phase
        
        itr       : int
                    number of iterations of ADMM algorithm
        
        verbose   : bool
                    option to display progress of the computation
        
        use_gpu   : bool
                    option to use gpu or not
        
        gpu_id    : int
                    number refering to which gpu will be used
    
    Returns
    -------
        f_real    : numpy.ndarray
                    3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)
    '''
    
    N, M, N_defocus = S0_stack.shape
    
    Dx = np.zeros((N, M, N_defocus)); Dx[0,0,0] = 1; Dx[0,-1,0] = -1;
    Dy = np.zeros((N, M, N_defocus)); Dy[0,0,0] = 1; Dy[-1,0,0] = -1;
    Dz = np.zeros((N, M, N_defocus)); Dz[0,0,0] = 1; Dz[0,0,-1] = -1;
    
    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        S0_stack_f = cp.fft.fftn(cp.array(S0_stack.astype('float32')), axes=(0,1,2))
        H_eff = cp.array(H_eff.astype('complex64'))
        
        Dx = cp.fft.fftn(cp.array(Dx),axes=(0,1,2))
        Dy = cp.fft.fftn(cp.array(Dy),axes=(0,1,2))
        Dz = cp.fft.fftn(cp.array(Dz),axes=(0,1,2))

        rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy + cp.conj(Dz)*Dz)+reg_re
        AHA      = cp.abs(H_eff)**2 + rho_term
        b_vec    = S0_stack_f * cp.conj(H_eff)

        z_para = cp.zeros((3, N, M, N_defocus))
        u_para = cp.zeros((3, N, M, N_defocus))
        D_vec  = cp.zeros((3, N, M, N_defocus))




        for i in range(itr):
            v_para    = cp.fft.fftn(z_para - u_para, axes=(1,2,3))
            b_vec_new = b_vec + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1] + cp.conj(Dz)*v_para[2])


            f_real = cp.real(cp.fft.ifftn(b_vec_new / AHA, axes=(0,1,2)))

            D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)


            z_para = D_vec + u_para

            z_para = softTreshold(z_para, lambda_re/rho, use_gpu=True, gpu_id=gpu_id)

            u_para += D_vec - z_para

            if verbose:
                print('Number of iteration computed (%d / %d)'%(i+1,itr))

            if i == itr-1:
                f_real = cp.asnumpy(f_real)
    
    else:
        
        S0_stack_f = fftn(S0_stack, axes=(0,1,2))
        
        Dx = fftn(Dx,axes=(0,1,2));
        Dy = fftn(Dy,axes=(0,1,2));
        Dz = fftn(Dz,axes=(0,1,2));

        rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy + np.conj(Dz)*Dz)+reg_re
        AHA      = np.abs(H_eff)**2 + rho_term
        b_vec    = S0_stack_f * np.conj(H_eff)

        z_para = np.zeros((3, N, M, N_defocus))
        u_para = np.zeros((3, N, M, N_defocus))
        D_vec  = np.zeros((3, N, M, N_defocus))


        for i in range(itr):
            v_para    = fftn(z_para - u_para, axes=(1,2,3))
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
                
    return f_real


def Dual_variable_ADMM_TV_deconv_3D(AHA, b_vec, rho, lambda_re, lambda_im, itr, verbose, use_gpu=False, gpu_id=0):
    
    '''
    
    3D TV deconvolution to solve for phase and absorption with weak object transfer function
    
    ADMM formulation:
        
        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2
    
    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 3D numpy array (4 diagonal matrices)
                     | AHA[0]  AHA[1] |
                     | AHA[2]  AHA[3] |
                  
        b_vec      : list
                     measured intensity stored with a list of 2 3D numpy array (2 vectors)
                     | b_vec[0] |
                     | b_vec[1] |
                     
        rho        : float
                     ADMM rho parameter
        
        lambda_re  : float
                     TV regularization parameter for phase
        
        lambda_im  : float
                     TV regularization parameter for absorption
        
        itr        : int
                     number of iterations of ADMM algorithm
        
        verbose    : bool
                     option to display progress of the computation
        
        use_gpu    : bool
                     option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        f_real     : numpy.ndarray
                     3D real scattering potential (unscaled phase) reconstruction with the size of (Ny, Nx, Nz)
                  
        f_imag     : numpy.ndarray
                     3D imaginary scattering potential (unscaled absorption) reconstruction with the size of (Ny, Nx, Nz)
    '''
    

    # ADMM deconvolution with anisotropic TV regularization
    
    N, M, L = b_vec[0].shape
    Dx = np.zeros((N, M, L)); Dx[0,0,0] = 1; Dx[0,-1,0] = -1;
    Dy = np.zeros((N, M, L)); Dy[0,0,0] = 1; Dy[-1,0,0] = -1;
    Dz = np.zeros((N, M, L)); Dz[0,0,0] = 1; Dz[0,0,-1] = -1;
    

    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Dx = cp.fft.fftn(cp.array(Dx));
        Dy = cp.fft.fftn(cp.array(Dy));
        Dz = cp.fft.fftn(cp.array(Dz));

        rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy + cp.conj(Dz)*Dz)

        z_para = cp.zeros((6, N, M, L))
        u_para = cp.zeros((6, N, M, L))
        D_vec = cp.zeros((6, N, M, L))


    else:
        Dx = fftn(Dx);
        Dy = fftn(Dy);
        Dz = fftn(Dz);

        rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy + np.conj(Dz)*Dz)

        z_para = np.zeros((6, N, M, L))
        u_para = np.zeros((6, N, M, L))
        D_vec = np.zeros((6, N, M, L))


    AHA[0] = AHA[0] + rho_term
    AHA[3] = AHA[3] + rho_term

    determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]


    for i in range(itr):


        if use_gpu:

            v_para = cp.fft.fftn(z_para - u_para,axes=(1,2,3))
            b_vec_new = [b_vec[0] + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1] + cp.conj(Dz)*v_para[2]),\
                         b_vec[1] + rho*(cp.conj(Dx)*v_para[3] + cp.conj(Dy)*v_para[4] + cp.conj(Dz)*v_para[5])]

            f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(AHA, b_vec_new, determinant=determinant, \
                                                                     use_gpu=use_gpu, gpu_id=gpu_id, move_cpu=not use_gpu)
            

            D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)
            D_vec[3] = f_imag - cp.roll(f_imag, -1, axis=1)
            D_vec[4] = f_imag - cp.roll(f_imag, -1, axis=0)
            D_vec[5] = f_imag - cp.roll(f_imag, -1, axis=2)


            z_para = D_vec + u_para

            z_para[:3,:,:] = softTreshold(z_para[:3,:,:], lambda_re/rho, use_gpu=True, gpu_id=gpu_id)
            z_para[3:,:,:] = softTreshold(z_para[3:,:,:], lambda_im/rho, use_gpu=True, gpu_id=gpu_id)

            u_para += D_vec - z_para

            if i == itr-1:
                f_real  = cp.asnumpy(f_real)
                f_imag = cp.asnumpy(f_imag)




        else:
            
            v_para = fftn(z_para - u_para,axes=(1,2,3))
            b_vec_new = [b_vec[0] + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1] + np.conj(Dz)*v_para[2]),\
                         b_vec[1] + rho*(np.conj(Dx)*v_para[3] + np.conj(Dy)*v_para[4] + np.conj(Dz)*v_para[5])]

            
            f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(AHA, b_vec_new, determinant=determinant)
            
            
            D_vec[0] = f_real - np.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - np.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - np.roll(f_real, -1, axis=2)
            D_vec[3] = f_imag - np.roll(f_imag, -1, axis=1)
            D_vec[4] = f_imag - np.roll(f_imag, -1, axis=0)
            D_vec[5] = f_imag - np.roll(f_imag, -1, axis=2)
            
            z_para = D_vec + u_para

            z_para[:3,:,:] = softTreshold(z_para[:3,:,:], lambda_re/rho)
            z_para[3:,:,:] = softTreshold(z_para[3:,:,:], lambda_im/rho)

            u_para += D_vec - z_para

        if verbose:
            print('Number of iteration computed (%d / %d)'%(i+1,itr))

    return f_real, f_imag


def cylindrical_shell_local_orientation(VOI, ps, psz, scale, beta=0.5, c_para=0.5, evec_idx = 0):
    
    '''
    
    segmentation of 3D cylindrical shell structure and the estimation of local orientation of the geometry
    
    Parameters
    ----------
        VOI      : numpy.ndarray
                   3D volume of interest
                    
        ps       : float
                   transverse pixel size of the image space
                          
        psz      : float
                   axial step size of the image space
                    
        scale    : list
                   list of feature size to scan through for a segmentation including multi-scale feature size
                
        beta     : float
                   value to control whether the segmentation need to highlight more or less on the shell-like feature
                   larger  -> highlight the most strong shell-like feature (more sparse segmentation) 
                   smaller -> highlight somewhat shell-like feature (more connected segmentation)
        
        c_para   : float
                   value to control whether the segmentation need to highlight more or less on the structure with overall large gradient
                 
        evec_idx : int
                   the index of eigenvector we consider for local orientation
                   0: smallest eigenvector, which cooresponds to the local orientation along the cylindrical shell
                   2: largest eigenvector, which cooresponds to the local orientation normal to the cylindrical shell
        
        
    Returns
    -------
        azimuth  : numpy.ndarray
                   the azimuthal angle of the computed local orientation
        
        theta    : numpy.ndarray
                   the theta part of the computed local orientation
        
        V_func   : numpy.ndarray
                   the segmentation map of the cylindrical shell structure
                   
        kernel   : numpy.ndarray
                   the kernel corresponding to the highlighting feature sizes in the segmentation with the size of (N_scale, Ny, Nx, Nz)
    
    '''
    
    # Hessian matrix filtering

    N, M, L = VOI.shape

    x_r = (np.r_[:M]-M//2)*ps
    y_r = (np.r_[:N]-N//2)*ps
    z_r = (np.r_[:L]-L//2)*psz


    xx_r, yy_r, zz_r = np.meshgrid(x_r,y_r,z_r)

    fx_r = ifftshift((np.r_[:M]-M//2)/ps/M)
    fy_r = ifftshift((np.r_[:N]-N//2)/ps/N)
    fz_r = ifftshift((np.r_[:L]-L//2)/psz/L)
    fxx_r, fyy_r, fzz_r = np.meshgrid(fx_r,fy_r,fz_r)

    diff_filter = np.zeros((3, N, M, L), complex)
    diff_filter[0] = 1j*2*np.pi*fxx_r
    diff_filter[1] = 1j*2*np.pi*fyy_r
    diff_filter[2] = 1j*2*np.pi*fzz_r

    V_func = np.zeros_like(VOI)
    kernel = np.zeros((len(scale),)+(N,M,L))


    t0 = time.time()

    for i,s in enumerate(scale):
        
        kernel[i] = np.exp(-(xx_r**2 + yy_r**2 + zz_r**2)/2/s**2)/(2*np.pi*s**2)**(3/2)
        Gaussian_3D_f = fftn(ifftshift(kernel[i]))*(ps*ps*psz)


        VOI_filtered = np.zeros((3,3,N,M,L))


        for p in range(3):
            for q in range(3):
                Hessian_filter = ((s)**2)*Gaussian_3D_f*diff_filter[p]*diff_filter[q]
                VOI_filtered[p,q] = np.real(ifftn(fftn(VOI)*Hessian_filter)/(ps*ps*psz))

        eigen_val, eigen_vec = np.linalg.eig(np.transpose(VOI_filtered,(2,3,4,0,1)))
        
        eig_val_idx = np.zeros((3,N,M,L))
        for p in range(3):
            eig_val_idx[p] = np.argpartition(np.abs(eigen_val), p, axis=3)[:,:,:,p]
        
        eig_val_sort = np.zeros((3, N, M, L), complex)
        

        for p in range(3):
            for q in range(3):
                eig_val_sort[q,eig_val_idx[q]==p] = eigen_val[eig_val_idx[q]==p,p]


        RB = np.abs(eig_val_sort[2]) / np.sqrt(np.abs(eig_val_sort[0])*np.abs(eig_val_sort[1]))
        S = np.sqrt(np.sum(np.abs(eig_val_sort)**2, axis=0))

        c = c_para*np.max(S)

        V_func_temp = (1-np.exp(-RB**2/2/beta**2))*(1-np.exp(-S**2/2/c**2))
        V_func_temp[np.real(eig_val_sort[2])>0] = 0

        if i ==0:
            V_func = V_func_temp.copy()
            orientation_vec = np.zeros((N, M, L, 3))

            for p in range(3):
                orientation_vec[eig_val_idx[evec_idx]==p] = np.real(eigen_vec[eig_val_idx[evec_idx]==p,:,p])
        else:
            larger_V_idx = (V_func_temp>V_func)
            V_func[larger_V_idx] = V_func_temp[larger_V_idx]

            for p in range(3):
                orientation_vec[np.logical_and(larger_V_idx,eig_val_idx[evec_idx]==p)] = np.real(eigen_vec[np.logical_and(larger_V_idx,eig_val_idx[evec_idx]==p),:,p])

        print('Finish V_map computation for scale = %.2f, elapsed time: %.2f'% (s, time.time()-t0))




    orientation_vec = np.transpose(orientation_vec, (3,0,1,2))
    
#     orientation_vec[0] = orientation_vec[0]*ps
#     orientation_vec[1] = orientation_vec[1]*ps
#     orientation_vec[2] = orientation_vec[2]*psz
    
    norm = np.sqrt(np.sum(np.abs(orientation_vec)**2,axis=0))
    theta = np.arccos(np.clip(orientation_vec[2]/norm,-1,1))
    azimuth = np.arctan2(orientation_vec[1], orientation_vec[0])
    azimuth = azimuth%(2*np.pi)
    theta[azimuth>np.pi] = np.pi-theta[azimuth>np.pi]
    azimuth[azimuth>np.pi] = azimuth[azimuth>np.pi] - np.pi

    print('Finish local orientation extraction, elapsed time:' + str(time.time()-t0))
    
    return azimuth, theta, V_func, kernel