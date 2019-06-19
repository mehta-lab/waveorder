import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from matplotlib.colors import hsv_to_rgb



def image_stack_viewer(image_stack,size=(10,10)):
    
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
        plt.imshow(image_stack[stack_idx],cmap='gray', vmin=min_val, vmax=max_val)
        plt.colorbar()

    def interact_plot_4D(stack_idx_1, stack_idx_2):    
        plt.figure(figsize=size)
        plt.imshow(image_stack[stack_idx_1,stack_idx_2],cmap='gray', vmin=min_val, vmax=max_val)
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


def plot_multicolumn(image_stack, num_col =2, size=10, set_title = False, titles=[]):
    
    '''
    
    Plot images in multiple columns
    
    Input: 
        image_stack : image stack in the size of (N_stack, N, M)
        num_col     : number of columns you wish to display
        size        : the size of one figure panel
        set_title   : Options for setting up titles of the figures
        titles      : titles for the figures
    
    '''
    
    N_stack, _, _ = image_stack.shape
    num_row = np.int(np.ceil(N_stack/num_col))
    figsize = (num_col*size, num_row*size)
    
    
    f1,ax = plt.subplots(num_row, num_col, figsize=figsize)
    
    if num_row == 1:
        for i in range(N_stack):
            col_idx = np.mod(i, num_col)
            ax1 = ax[col_idx].imshow(image_stack[i], cmap='gray')
            plt.colorbar(ax1,ax=ax[col_idx])
            
            if set_title == True:
                ax[col_idx].set_title(titles[i])
    else: 
        for i in range(N_stack):
            row_idx = i//num_col
            col_idx = np.mod(i, num_col)
            ax1 = ax[row_idx, col_idx].imshow(image_stack[i], cmap='gray')
            plt.colorbar(ax1,ax=ax[row_idx, col_idx])
            
            if set_title == True:
                ax[row_idx, col_idx].set_title(titles[i])


def plot_hsv(image_stack, max_val=1, size=5):
    
    N_channel = len(image_stack)
    
    if N_channel == 2:
        I_hsv = np.transpose(np.stack([image_stack[0]/np.max(image_stack[0]), \
                                       np.ones_like(image_stack[0]), \
                                       np.minimum(1, image_stack[1]/np.max(image_stack[1])/max_val)]), (1,2,0))
        I_rgb = hsv_to_rgb(I_hsv)
        
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