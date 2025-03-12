#%% Imports -------------------------------------------------------------------

import nd2
import numpy as np

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.exposure import adjust_gamma

#%% Function : format_stack() -------------------------------------------------

def format_stack(path, rf=0.5):
    
    with nd2.ND2File(path) as ndfile:
        
        # voxSize
        voxsize0 = (
            ndfile.voxel_size()[2],
            ndfile.voxel_size()[1],
            ndfile.voxel_size()[0],
            )
        
        # Determine isotropic rescaling factor (rfi)
        rfi = voxsize0[1] / voxsize0[0]
        
        # Load & rescale stack
        stk = ndfile.asarray()
        stk = rescale(stk, (1, 1, rfi, rfi), order=0) # iso rescale (rfi)
        stk = rescale(stk, (rf, 1, rf, rf), order=0) # custom rescale (rf)
            
        # Flip z axis
        stk = np.flip(stk, axis=0)
        
        # Adjust voxSize
        voxsize = voxsize0[0] / rf
        
    return stk, voxsize

#%% Function : merge_stack() --------------------------------------------------

def merge_stack(stk, voxsize):
    C1, C3, C4 = stk[:, 0, ...], stk[:, 2, ...], stk[:, 3, ...]
    C1 = adjust_gamma(norm_pct(C1), gamma=0.5)
    C3 = adjust_gamma(norm_pct(C3), gamma=0.5)
    C4 = adjust_gamma(norm_pct(C4), gamma=0.5)
    return (C1 + C3 + C4 ) / 3
