import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import uniform_filter


import re
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



def genStarTarget(N, M, blur_px = 2):
    
    '''
    
    generate Siemens star for simulation target
    
    Input:
        (N, M)  : (y, x) dimension of the simulated image
        blur_px : degree of the blurring imposed on the generated image
        
    Output:
        star    : Siemens star with the size of (N, M)
        theta   : polar angle np array with the size of (N, M)
    
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
    star = np.pad(star[60:-60,60:-60],(60,),mode='constant')
    star[star<1] = 0
    
    # Filter to prevent aliasing


    Gaussian = np.exp(-rho**2/(2*blur_px**2))
    
    star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))
    # star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))*(2+np.sin(2*np.pi*(1/5)*rho))
    star /= np.max(star)
    
    
    return star, theta, xx

def gen_sphere_target(img_dim, ps, psz, radius, blur_size = 0.1):
    
    
    N,M,L = img_dim
    x = (np.r_[:N]-N//2)*ps
    y = (np.r_[:M]-M//2)*ps
    z = (np.r_[:L]-L//2)*psz
    
    xx, yy, zz = np.meshgrid(x,y,z)
    
    rho = np.sqrt(xx**2 + yy**2 + zz**2)

    sphere = np.zeros_like(xx)
    sphere[xx**2 + yy**2 + zz**2<radius**2] = 1
    
    Gaussian = np.exp(-rho**2/(2*blur_size**2))
    
    sphere = np.maximum(0, np.real(ifftn(fftn(sphere,axes=(0,1,2)) * \
                                         fftn(ifftshift(Gaussian),axes=(0,1,2)),axes=(0,1,2))))
    sphere /= np.max(sphere)
    
    
    return sphere




def gen_coordinate(img_dim, ps):
    
    '''
    
    generate spatial and spatial frequency coordinate arrays
    
    Input:
        img_dim : describes the (y, x) dimension of the computed 2D space
        ps      : pixel size
        
    Output:
        xx      : 2D x-coordinate array
        yy      : 2D y-coordinate array
        fxx     : 2D spatial frequency array in x-dimension
        fyy     : 2D spatial frequency array in y-dimension
    
    '''
    
    N, M = img_dim
    
    fx = ifftshift((np.r_[:M]-M/2)/M/ps)
    fy = ifftshift((np.r_[:N]-N/2)/N/ps)
    x  = ifftshift((np.r_[:M]-M/2)*ps)
    y  = ifftshift((np.r_[:N]-N/2)*ps)


    xx, yy = np.meshgrid(x, y)
    fxx, fyy = np.meshgrid(fx, fy)

    return (xx, yy, fxx, fyy)


def image_upsampling(Ic_image, upsamp_factor = 1, bg = 0, method=None):
    F = lambda x: ifftshift(fft2(fftshift(x)))
    iF = lambda x: ifftshift(ifft2(fftshift(x)))
    
    N_channel, Ncrop, Mcrop, N_defocus = Ic_image.shape

    N = Ncrop*upsamp_factor
    M = Mcrop*upsamp_factor

    Ic_image_up = np.zeros((N_channel,N,M,N_defocus))
    
    for i in range(0,N_channel):
        for j in range(0, N_defocus):
            if method == 'BICUBIC':
                Ic_image_up[i,:,:,j] = np.array(Image.fromarray(Ic_image[i,:,:,j]-bg).resize((M,N), Image.BICUBIC))
            else:
                Ic_image_up[i,:,:,j] = abs(iF(np.pad(F(np.maximum(0,Ic_image[i,:,:,j]-bg)),\
                                      (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))
            
        
    return Ic_image_up

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

def softTreshold(x, threshold, use_gpu=False):
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        magnitude = cp.abs(x)
        ratio = cp.maximum(0, magnitude-threshold) / magnitude
    else:
        magnitude = np.abs(x)
        ratio = np.maximum(0, magnitude-threshold) / magnitude
    
    x_threshold = x*ratio
    
    return x_threshold


def uniform_filter_2D(image, size, use_gpu=False):
    
    N, M = image.shape
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        
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

        filtered_xy = cp.real(cp.fft.ifft(cp.fft.fft(image_bound_x,axis=1)*kernel_x[cp.newaxis,:],axis=1))
        filtered_xy = filtered_xy[:,M:2*M]
    else:
        filtered_xy = uniform_filter(image, size=size)
        
        
    return filtered_xy
