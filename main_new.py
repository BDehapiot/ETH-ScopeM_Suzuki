#%% Imports -------------------------------------------------------------------

import nd2
import time
import shutil
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.filters.rank import median
from skimage.morphology import ball, h_maxima

#%% Comments ------------------------------------------------------------------

'''
- C1 : NRP2-EGFP (protein of interest)
- C2 : AF594-labelled virus
- C3 : EEA1 (early endosome marker)
- C4 : Hoechst (nucleus)
'''

#%% Inputs --------------------------------------------------------------------

# Procedure
overwrite = {
    "preprocess" : 0,
    }

# Parameters
rf = 0.5

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
paths = list(data_path.glob("*.nd2"))

#%% Function : preprocess() ---------------------------------------------------

def preprocess(path, rf=0.5):
        
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
        voxsize1 = (
            voxsize0[0] / rf, 
            voxsize0[1] / (rfi * rf),
            voxsize0[2] / (rfi * rf),
            )
        
        # Setup directory 
        dir_path = Path(data_path / path.stem)
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        dir_path.mkdir(exist_ok=True)
        
        # Save       
        stk_name = path.stem + f"_{voxsize1[0]}_stk.tif"
        resolution = (1 / voxsize1[1], 1 / voxsize1[2])
        metadata = {"axes" : "ZCYX", "spacing" : voxsize1[0], "unit" : "um"}
        tifffile.imwrite(
            dir_path / stk_name, stk, 
            imagej=True, resolution=resolution, metadata=metadata
            )

#%% Function : process() ------------------------------------------------------

def process(path):

    # Load data
    dir_path = Path(data_path / path.stem)
    stk_path = list(dir_path.glob("*_stk.tif"))[0]
    stk = io.imread(stk_path)
        
    return stk

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Preprocess --------------------------------------------------------------

    # Paths
    tmp_paths = []
    for path in paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_stk.tif"))
        if not stk_path or not stk_path[0] or overwrite["preprocess"]:
            tmp_paths.append(path)
    
    print("preprocess : ", end="", flush=True)
    t0 = time.time()
    
    # Execute
    Parallel(n_jobs=-1)(
        delayed(preprocess)(path, rf=rf) 
        for path in tmp_paths
        )  
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Process -----------------------------------------------------------------

    print("process    : ", end="", flush=True)
    t0 = time.time()
    
    # # Execute
    # Parallel(n_jobs=-1)(
    #     delayed(process)(path) 
    #     for path in paths
    #     )  
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
#%%

    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects    
    
    # Load data
    stk = process(paths[4])
    # voxsize = 
    
    # Format data
    cyt = (norm_pct(stk[..., 0]) * 255).astype("uint8")
    ncl = (norm_pct(stk[..., 3]) * 255).astype("uint8")
    cyt = median(cyt, footprint=ball(5))
    ncl = median(ncl, footprint=ball(5))
    cyt_thresh = threshold_otsu(cyt)
    ncl_thresh = threshold_otsu(ncl)
    cyt_msk = cyt > cyt_thresh
    ncl_msk = ncl > ncl_thresh
    cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
    ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
    
    # Display
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(
        cyt, name="cyt", visible=1,
        blending="additive", colormap="bop orange",
        )
    viewer.add_image(
        ncl, name="nuclei", visible=1,
        blending="additive", colormap="bop blue",
        )
    viewer.add_image(
        cyt_msk, name="cyt_msk", visible=0,
        rendering="attenuated_mip", attenuation=0.5, colormap="bop orange",
        )
    viewer.add_image(
        ncl_msk, name="ncl_msk", visible=0,
        rendering="attenuated_mip", attenuation=0.5, colormap="bop blue",
        )