#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from pathlib import Path

# functions
from functions import import_htk, prepare_htk, save_tif 

#%% Inputs --------------------------------------------------------------------

voxsize = 0.2
nSlices = 5
tags_out = ["Cla__"]

#%% Initialize ----------------------------------------------------------------

# data_path = Path("D:\local_Suzuki\data")
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data")
train_path = Path("data", "train")
htk_paths = list(data_path.rglob("*.nd2"))   

#%% Function(s) ---------------------------------------------------------------

def extract(paths, voxsize=0.2, nSlices=5):

    for i, path in enumerate(paths):
                
        if any(tag not in path.stem for tag in tags_out):
        
            print("\n" + "load : " + path.name)    
        
            # Import and prepare htk
            _, htk = import_htk(path, voxsize=voxsize)
            if htk.shape[1] != 4:
                continue
            prp = prepare_htk(htk)
            
            # Save random slices
            idxs = np.random.choice(
                np.arange(prp.shape[0]), size=nSlices, replace=False)   
            for idx in idxs:
                suffix = f"{voxsize}_z{idx:02d}"
                save_name = path.stem + "_" + suffix + ".tif"
                save_path = train_path / save_name
                print(suffix, end=", ", flush=False)
                save_tif(prp[idx, ...], save_path, voxsize=voxsize)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    extract(htk_paths, voxsize=voxsize, nSlices=nSlices)
