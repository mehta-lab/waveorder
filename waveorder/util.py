import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
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
    
    Hz_stack = np.zeros((N_stack, N, M), complex)
    fr = (fxx**2 + fyy**2)**(1/2)
    
    for i in range(N_stack):
        Hz_stack[i] = Pupil_support * np.exp(1j*2*np.pi/lambda_in*z_stack[i]* ((1 - lambda_in**2 * fr**2) * Pupil_support)**(1/2))

    
    return Hz_stack


def image_upsampling(Ic_image, upsamp_factor = 1, bg = 0, method=None):
    F = lambda x: ifftshift(fft2(fftshift(x)))
    iF = lambda x: ifftshift(ifft2(fftshift(x)))
    
    N_defocus, Nimg, Ncrop, Mcrop = Ic_image.shape

    N = Ncrop*upsamp_factor
    M = Mcrop*upsamp_factor

    Ic_image_up = np.zeros((N_defocus,Nimg,N,M))
    
    for i in range(0,Nimg):
        for j in range(0, N_defocus):
            if method == 'BICUBIC':
                Ic_image_up[j,i] = np.array(Image.fromarray(Ic_image[j,i]-bg).resize((M,N), Image.BICUBIC))
            else:
                Ic_image_up[j,i] = abs(iF(np.pad(F(np.maximum(0,Ic_image[j,i]-bg)),\
                                      (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))
            
        
    return Ic_image_up

def softTreshold(x, threshold):
    
    magnitude = np.abs(x)
    ratio = np.maximum(0, magnitude-threshold) / magnitude
    
    x_threshold = x*ratio
    
    return x_threshold