#%% Imports -------------------------------------------------------------------

import nd2
import time
import napari
import numpy as np
from pathlib import Path

# skimage
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.feature import blob_log
# from 

#%% Comments ------------------------------------------------------------------

'''
- C1 : NRP2-EGFP (protein of interest)
- C2 : AF594-labelled virus
- C3 : EEA1 (early endosome marker)
- C4 : Hoechst (nucleus)
    
'''

#%% Inputs --------------------------------------------------------------------

# Parameters
rf = 0.25
sigma = 2

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

def downscale(stk, rf=0.5, order=0):
    return rescale(stk, (1, rf, rf), order=order)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Path
    path = stk_paths[0]
    
    # # Load data
    with nd2.ND2File(path) as ndfile:
        
        # Read metadata
        nZ, nC, nY, nX = ndfile.shape
        voxSize = (
            ndfile.voxel_size()[2] * rf,
            ndfile.voxel_size()[1] * rf,
            ndfile.voxel_size()[0],
            )

        # Open data
        tmp = ndfile.asarray()
        C1 = downscale(tmp[:, 0, ...], rf=rf, order=0)
        C2 = downscale(tmp[:, 1, ...], rf=rf, order=0)
        C3 = downscale(tmp[:, 2, ...], rf=rf, order=0)
        C4 = downscale(tmp[:, 3, ...], rf=rf, order=0)
        C1 = np.flip(C1, axis=0)
        C2 = np.flip(C2, axis=0)
        C3 = np.flip(C3, axis=0)
        C4 = np.flip(C4, axis=0)
        
        # Spot detection
        C2_blob = blob_log(
            C2, min_sigma=0.5, max_sigma=5, num_sigma=5, threshold=0.01)
        
        # # Display
        # viewer = napari.Viewer()
        # scale = [voxSize[0] / voxSize[1], 1, 1]
        # viewer.add_image(
        #     C2, name="virus", scale=scale, visible=1,
        #     blending="additive", colormap="magenta",
        #     )
        # viewer.add_image(
        #     C2_process, name="virus", scale=scale, visible=1,
        #     blending="additive", colormap="magenta",
        #     )
        
        # # Display
        # viewer = napari.Viewer()
        # scale = [voxSize[0] / voxSize[1], 1, 1]
        # viewer.add_image(
        #     C1, name="NRP2", scale=scale, visible=1,
        #     blending="additive", colormap="green",
        #     )
        # viewer.add_image(
        #     C2, name="virus", scale=scale, visible=1,
        #     blending="additive", colormap="magenta",
        #     )
        # viewer.add_image(
        #     C3, name="EEA1", scale=scale, visible=0,
        #     blending="additive", colormap="bop orange",
        #     )
        # viewer.add_image(
        #     C4, name="nuclei", scale=scale, visible=1,
        #     blending="additive", colormap="bop blue",
        #     )
    
    pass