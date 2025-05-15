#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from pathlib import Path

# functions
from functions import check_nd2, import_nd2, prepare_data, save_tif 

#%% Inputs --------------------------------------------------------------------

voxsize = 0.2
nSlices = 3

#%% Initialize ----------------------------------------------------------------

# Paths 
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data")
train_path = Path("data", "train")

#%% Function(s) ---------------------------------------------------------------

def extract(data_path, train_path, voxsize=0.2, nSlices=5):
    
    # Nested function(s) ------------------------------------------------------
    
    def _extract(nd2_path, z_idx, voxsize=0.2):
        
        # Extract & prepare images
        _, C1 = import_nd2(nd2_path, z=z_idx, c=0, voxsize=voxsize)
        _, C4 = import_nd2(nd2_path, z=z_idx, c=3, voxsize=voxsize)
        prp = prepare_data(C1, C4)
        
        # Save
        suffix = f"{voxsize}_z{z_idx:02d}"
        save_name = nd2_path.stem + "_" + suffix + ".tif"
        print(save_name)
        save_path = train_path / save_name
        save_tif(prp, save_path, voxsize=voxsize)
        
    # Execute -----------------------------------------------------------------

    # Paths
    nd2_paths = list(data_path.rglob("*.nd2"))   

    # Extract & prepare images
    for nd2_path in nd2_paths:

        # Check nd2 file
        shape = check_nd2(nd2_path)
        if shape[1] != 4:
            continue
        
        # Random z_idx
        z_idxs = np.random.choice(
            np.arange(shape[0]), size=nSlices, replace=False)  
        
        for z_idx in z_idxs:
            _extract(nd2_path, z_idx, voxsize=voxsize)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    extract(data_path, train_path, voxsize=voxsize, nSlices=nSlices)

#%%

    # # Extract images for already existing masks
    # msk_paths = list(train_path.glob("*_mask.tif"))
    # for msk_path in msk_paths:
    #     name = msk_path.stem[:-13] + ".nd2"
    #     z_idx = int(msk_path.stem[-7:-5])
    #     nd2_path = list(data_path.rglob(f"*{name}"))[0]
    #     _extract(nd2_path, -(z_idx // 2), voxsize=voxsize)    
