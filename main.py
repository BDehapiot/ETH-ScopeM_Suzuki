#%% Imports -------------------------------------------------------------------

import nd2
import time
import napari
import numpy as np
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.morphology import ball, h_maxima

#%% Comments ------------------------------------------------------------------

'''
- C1 : NRP2-EGFP (protein of interest)
- C2 : AF594-labelled virus
- C3 : EEA1 (early endosome marker)
- C4 : Hoechst (nucleus)
    
'''

#%% Inputs --------------------------------------------------------------------

# Parameters
rf = 0.5

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

def format_stack(path, rf=0.5):
        
    with nd2.ND2File(path) as ndfile:
        
        # voxSize
        voxSize0 = (
            ndfile.voxel_size()[2],
            ndfile.voxel_size()[1],
            ndfile.voxel_size()[0],
            )
        
        # Determine isotropic rescaling factor (rfi)
        rfi = voxSize0[1] / voxSize0[0]
        
        # Load & rescale stack
        stk = ndfile.asarray()
        stk = rescale(stk, (1, 1, rfi, rfi), order=0) # iso rescale (rfi)
        stk = rescale(stk, (rf, 1, rf, rf), order=0) # custom rescale (rf)
            
        # Flip z axis
        stk = np.flip(stk, axis=0)
        
        # Split channels
        C1, C2, C3, C4 = (
            stk[:, 0, ...], stk[:, 1, ...], stk[:, 2, ...], stk[:, 3, ...]) 
        
        # Adjust voxSize
        voxSize1 = (
            voxSize0[0] / rf, 
            voxSize0[1] / (rfi * rf),
            voxSize0[2] / (rfi * rf),
            )
    
    return C1, C2, C3, C4, voxSize1

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Path
    path = stk_paths[2]
    print(f"{path.name}")
    
    # -------------------------------------------------------------------------
    
    # Format stack
    print("format_stack() : ", end="", flush=True)
    t0 = time.time()
    C1, C2, C3, C4, voxSize = format_stack(path, rf=rf)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # # Display
    # viewer = napari.Viewer()
    # scale = [voxSize[0] / voxSize[1], 1, 1]
    # viewer.add_image(
    #     C1, name="NRP2", scale=scale, visible=1,
    #     blending="additive", colormap="magenta",
    #     )
    # viewer.add_image(
    #     C2, name="virus", scale=scale, visible=1,
    #     blending="additive", colormap="green",
    #     )
    # viewer.add_image(
    #     C3, name="EEA1", scale=scale, visible=0,
    #     blending="additive", colormap="bop orange",
    #     )
    # viewer.add_image(
    #     C4, name="nuclei", scale=scale, visible=1,
    #     blending="additive", colormap="bop blue",
    #     )
    
    # -------------------------------------------------------------------------
    
    from skimage.filters.rank import median
    from skimage.morphology import remove_small_objects
    
    # Cytoplasm mask
    tmp_cyto = C1
    tmp_cyto = (norm_pct(tmp_cyto) * 255).astype("uint8")
    tmp_cyto = median(tmp_cyto, footprint=ball(5))
    msk_cyto = tmp_cyto > 20
    msk_cyto = remove_small_objects(msk_cyto, min_size=1e5)
    
    # Nuclei mask
    tmp_nuclei = C4
    tmp_nuclei = (norm_pct(tmp_nuclei) * 255).astype("uint8")
    tmp_nuclei = median(tmp_nuclei, footprint=ball(5))
    msk_nuclei = tmp_nuclei > 50
    
    # Display
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    scale = [voxSize[0] / voxSize[1], 1, 1]
    viewer.add_image(
        tmp_cyto, name="cytoplasm", scale=scale, visible=1,
        blending="additive", colormap="bop orange",
        )
    viewer.add_image(
        tmp_nuclei, name="nuclei", scale=scale, visible=1,
        blending="additive", colormap="bop blue",
        )
    viewer.add_image(
        msk_cyto, name="cytoplasm mask", scale=scale, visible=0,
        rendering="attenuated_mip", attenuation=0.5, colormap="bop orange",
        )
    viewer.add_image(
        msk_nuclei, name="nuclei mask", scale=scale, visible=0,
        rendering="attenuated_mip", attenuation=0.5, colormap="bop blue",
        )
    
    # -------------------------------------------------------------------------
    
    # # Detect local maxima
    # print("local_maxima() : ", end="", flush=True)
    # t0 = time.time()
    
    # C2_lmax = h_maxima(C2.astype(float), 50, footprint=ball(2 * rf))
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Display
    # viewer = napari.Viewer()
    # scale = [voxSize[0] / voxSize[1], 1, 1]
    # viewer.add_image(
    #     C2, name="virus", scale=scale, visible=1,
    #     blending="additive", colormap="green",
    #     contrast_limits=[0, 1000],
    #     )
    # viewer.add_image(
    #     C2_lmax, name="virus", scale=scale, visible=1,
    #     blending="additive", colormap="gray",
    #     contrast_limits=[0, 1],
    #     )


    
    pass