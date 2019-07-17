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

    star = 1 + np.cos(40*theta)
    star = np.pad(star[10:-10,10:-10],(10,),mode='constant')

    # Filter to prevent aliasing


    Gaussian = np.exp(-rho**2/(2*blur_px**2))

    star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))
    star /= np.max(star)
    
    
    return star, theta

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



def gen_Pupil(fxx, fyy, NA, lambda_in):
    
    N, M = fxx.shape
    
    Pupil = np.zeros((N,M))
    fr = (fxx**2 + fyy**2)**(1/2)
    Pupil[ fr < NA/lambda_in] = 1
    
    return Pupil


def gen_Hz_stack(fxx, fyy, Pupil_support, lambda_in, z_stack):
    
    N, M = fxx.shape
    N_stack = len(z_stack)
    N_defocus = len(z_stack)
    
    fr = (fxx**2 + fyy**2)**(1/2)
    
    oblique_factor = ((1 - lambda_in**2 * fr**2) *Pupil_support)**(1/2) / lambda_in
    
    Hz_stack = Pupil_support[:,:,np.newaxis] * np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:]*\
                                                      oblique_factor[:,:,np.newaxis])
    G_fun_z = -1j/4/np.pi* Pupil_support[:,:,np.newaxis] * np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:] * \
                                                                 oblique_factor[:,:,np.newaxis]) /(oblique_factor[:,:,np.newaxis]+1e-15)

    
    return Hz_stack, G_fun_z


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

def softTreshold(x, threshold):
    
    magnitude = np.abs(x)
    ratio = np.maximum(0, magnitude-threshold) / magnitude
    
    x_threshold = x*ratio
    
    return x_threshold