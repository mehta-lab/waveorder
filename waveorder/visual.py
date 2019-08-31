import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import uniform_filter



def image_stack_viewer(image_stack,size=(10,10), colormap='gray'):
    
    '''
    
    Visualize 3D and 4D image stack interactively in jupyter notebook
    
    Input: 
        image_stack : a 3D or 4D numpy array with size of (N_stack, N, M) or (N_stack, Nchannel, N, M)
        size        : the dimension of the figure panel (width, height)
    
    '''
    
    max_val = np.max(image_stack)
    min_val = np.min(image_stack)
    
    def interact_plot_3D(stack_idx):    
        plt.figure(figsize=size)
        plt.imshow(image_stack[stack_idx],cmap=colormap, vmin=min_val, vmax=max_val)
        plt.colorbar()

    def interact_plot_4D(stack_idx_1, stack_idx_2):    
        plt.figure(figsize=size)
        plt.imshow(image_stack[stack_idx_1,stack_idx_2],cmap=colormap, vmin=min_val, vmax=max_val)
        plt.colorbar()
    
    
    if image_stack.ndim == 3:    
        return interact(interact_plot_3D, stack_idx=widgets.IntSlider(value=0, min=0, max=len(image_stack)-1, step=1))
    else:
        return interact(interact_plot_4D, stack_idx_1=widgets.IntSlider(value=0, min=0, max=image_stack.shape[0]-1, step=1),\
                       stack_idx_2=widgets.IntSlider(value=0, min=0, max=image_stack.shape[1]-1, step=1))
    
    
def hsv_stack_viewer(image_stack, max_val=1, size=5):
    
    '''
    
    '''
    image_stack1 = image_stack[0]
    image_stack2 = image_stack[1]
    
    N_stack = len(image_stack1)
    I_rgb = np.zeros(image_stack1.shape+(3,))
    
    
    for i in range(N_stack):
        I_hsv = np.transpose(np.stack([image_stack1[i]/np.max(image_stack1[i]), \
                                           np.ones_like(image_stack1[i]), \
                                           np.minimum(1, image_stack2[i]/np.max(image_stack2[i])/max_val)]), (1,2,0))
        I_rgb[i] = hsv_to_rgb(I_hsv)
        
    V, H = np.mgrid[0:1:500j, 0:1:500j]
    S = np.ones_like(V)
    HSV = np.dstack((V,S,H))
    RGB = hsv_to_rgb(HSV)
    
    def interact_plot_hsv(stack_idx):
    
        f1,ax = plt.subplots(1, 2, figsize=(size+size/2, size))
        ax[0].imshow(I_rgb[stack_idx])

        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("V")
        plt.ylabel("H")
        plt.title("$S_{HSV}=1$")

        plt.tight_layout()
    
       
    return interact(interact_plot_hsv, stack_idx=widgets.IntSlider(value=0, min=0, max=len(image_stack1)-1, step=1))

    
def parallel_4D_viewer(image_stack, num_col = 2, size=10):
    
    '''
    
    Simultaneous visualize all channels of image stack interactively in jupyter notebook
    
    Input: 
        image_stack : a 4D numpy array with size of (N_stack, Nchannel, N, M)
        num_col     : number of columns you wish to display
        size        : the size of one figure panel 
    
    '''
    
    
    max_val = np.max(image_stack)
    min_val = np.min(image_stack)
    
    N_stack, N_channel, _, _ = image_stack.shape
    num_row = np.int(np.ceil(N_channel/num_col))
    figsize = (num_col*size, num_row*size)
    
    def interact_plot(stack_idx):
        
        f1,ax = plt.subplots(num_row, num_col, figsize=figsize)
        if num_row == 1:
            for i in range(N_channel):
                col_idx = np.mod(i, num_col)
                ax1 = ax[col_idx].imshow(image_stack[stack_idx, i], cmap='gray')
                plt.colorbar(ax1,ax=ax[col_idx])
        else:
            for i in range(N_channel):
                row_idx = i//num_col
                col_idx = np.mod(i, num_col)
                ax1 = ax[row_idx, col_idx].imshow(image_stack[stack_idx, i], cmap='gray')
                plt.colorbar(ax1,ax=ax[row_idx, col_idx])
    
    return interact(interact_plot, stack_idx=widgets.IntSlider(value=0, min=0, max=N_stack-1, step=1))


def plot_multicolumn(image_stack, num_col =2, size=10, set_title = False, titles=[], colormap='gray'):
    
    '''
    
    Plot images in multiple columns
    
    Input: 
        image_stack : image stack in the size of (N_stack, N, M)
        num_col     : number of columns you wish to display
        size        : the size of one figure panel
        set_title   : Options for setting up titles of the figures
        titles      : titles for the figures
    
    '''
    
    N_stack = len(image_stack)
    num_row = np.int(np.ceil(N_stack/num_col))
    figsize = (num_col*size, num_row*size)
    
    
    f1,ax = plt.subplots(num_row, num_col, figsize=figsize)
    
    if num_row == 1:
        for i in range(N_stack):
            col_idx = np.mod(i, num_col)
            ax1 = ax[col_idx].imshow(image_stack[i], cmap=colormap)
            plt.colorbar(ax1,ax=ax[col_idx])
            
            if set_title == True:
                ax[col_idx].set_title(titles[i])
    else: 
        for i in range(N_stack):
            row_idx = i//num_col
            col_idx = np.mod(i, num_col)
            ax1 = ax[row_idx, col_idx].imshow(image_stack[i], cmap=colormap)
            plt.colorbar(ax1,ax=ax[row_idx, col_idx])
            
            if set_title == True:
                ax[row_idx, col_idx].set_title(titles[i])


def plot_hsv(image_stack, max_val=1, size=5):
    
    N_channel = len(image_stack)
    
    if N_channel == 2:
        I_hsv = np.transpose(np.array([image_stack[0]/np.pi, \
                                       np.ones_like(image_stack[0]), \
                                       np.minimum(1, image_stack[1]/np.max(image_stack[1])/max_val)]), (1,2,0))
        I_rgb = hsv_to_rgb(I_hsv.copy())
        
        f1,ax = plt.subplots(1, 2, figsize=(size+size/2, size))
        ax[0].imshow(I_rgb)
        
        V, H = np.mgrid[0:1:500j, 0:1:500j]
        S = np.ones_like(V)
        HSV = np.dstack((V,S,H))
        RGB = hsv_to_rgb(HSV)
        
        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("V")
        plt.ylabel("H")
        plt.title("$S_{HSV}=1$")
        
        plt.tight_layout()

    
    else:
        raise("plot_hsv does not support N_channel >2 rendering")
        
        
def plot_phase_hsv(image_stack, max_val_V=1, max_val_S=1, size=5):
    
    N_channel = len(image_stack)
    
    if N_channel == 3:
        I_hsv = np.transpose(np.array([image_stack[0]/np.pi, \
                                       np.clip(image_stack[1]/np.max(image_stack[1])/max_val_S, 0, 1), \
                                       np.clip(image_stack[2]/np.max(image_stack[2])/max_val_V, 0, 1)]), (1,2,0))
        I_rgb = hsv_to_rgb(I_hsv.copy())
        
        f1,ax = plt.subplots(1, 2, figsize=(size+size/2, size))
        ax[0].imshow(I_rgb)
        
        V, H = np.mgrid[0:1:500j, 0:1:500j]
        S = np.ones_like(V)
        HSV = np.dstack((V,H,S))
        RGB = hsv_to_rgb(HSV)
        
        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("S")
        plt.ylabel("H")
        plt.title("$V_{HSV}=1$")
        
        plt.tight_layout()

    
    else:
        raise("plot_hsv does not support N_channel >3 rendering")
        
        
        
        
def plotVectorField(img,
                    orientation,
                    anisotropy=1,
                    spacing=20,
                    window=20,
                    linelength=20,
                    linewidth=3,
                    linecolor='g',
                    colorOrient=True,
                    cmapOrient='hsv',
                    threshold=None,
                    alpha=1,
                    clim=[None, None],
                    cmapImage='gray'):
    """Overlays orientation field on the image. Returns matplotlib image axes.
    Options:
        threshold:
        colorOrient: if it is True, then color the lines by their orientation.
        linelength : can be a scalar or an image the same size as the orientation.
    Parameters
    ----------
    img: nparray
        image to overlay orientation lines on
    orientation: nparray
        orientation in radian
    anisotropy: nparray
    spacing: int
    window: int
    linelength: int
        can be a scalar or an image the same size as the orientation
    linewidth: int
        width of the orientation line
    linecolor: str
    colorOrient: bool
        if it is True, then color the lines by their orientation.
    cmapOrient:
    threshold: nparray
        a binary numpy array, wherever the map is 0, ignore the plotting of the line
    alpha: int
        line transparency. [0,1]. lower is more transparent
    clim: list
        [min, max], min and max intensities for displaying img
    cmapImage:
        colormap for displaying the image
    Returns
    -------
    im_ax: obj
        matplotlib image axes
    """

    # plot vector field representaiton of the orientation map
    
    # Compute U, V such that they are as long as line-length when anisotropy = 1.
    U, V =  anisotropy*linelength * np.cos(2 * orientation), anisotropy*linelength * np.sin(2 * orientation)
    USmooth = uniform_filter(U, (window, window)) # plot smoothed vector field
    VSmooth = uniform_filter(V, (window, window)) # plot smoothed vector field
    azimuthSmooth = 0.5*np.arctan2(VSmooth,USmooth)
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    USmooth, VSmooth = RSmooth*np.cos(azimuthSmooth), RSmooth*np.sin(azimuthSmooth)

    nY, nX = img.shape
    Y, X = np.mgrid[0:nY,0:nX] # notice the reversed order of X and Y
    
    # Plot sparsely sampled vector lines
    Plotting_X = X[::spacing, ::spacing]
    Plotting_Y = Y[::spacing, ::spacing]
    Plotting_U = linelength * USmooth[::spacing, ::spacing]
    Plotting_V = linelength * VSmooth[::spacing, ::spacing]
    Plotting_R = RSmooth[::spacing, ::spacing]
    
    if threshold is None:
        threshold = np.ones_like(X) # no threshold
    Plotting_thres = threshold[::spacing, ::spacing]
    Plotting_orien = ((azimuthSmooth[::spacing, ::spacing])%np.pi)*180/np.pi
    
    
    if colorOrient:
        im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
        plt.title('Orientation map')
        plt.quiver(Plotting_X[Plotting_thres==1], Plotting_Y[Plotting_thres==1],
                   Plotting_U[Plotting_thres==1], Plotting_V[Plotting_thres==1], Plotting_orien[Plotting_thres==1],
                   cmap=cmapOrient,
                   edgecolor=linecolor,facecolor=linecolor,units='xy', alpha=alpha, width=linewidth,
                   headwidth = 0, headlength = 0, headaxislength = 0,
                   scale_units = 'xy',scale = 1, angles = 'uv', pivot = 'mid')
    else:
        im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
        plt.title('Orientation map')
        plt.quiver(Plotting_X[Plotting_thres==1], Plotting_Y[Plotting_thres==1],
                   Plotting_U[Plotting_thres==1], Plotting_V[Plotting_thres==1],
                   edgecolor=linecolor,facecolor=linecolor,units='xy', alpha=alpha, width=linewidth,
                   headwidth = 0, headlength = 0, headaxislength = 0,
                   scale_units = 'xy',scale = 1, angles = 'uv', pivot = 'mid')

    return im_ax
        
        

