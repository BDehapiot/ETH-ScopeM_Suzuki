#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# functions
from functions import format_stack, prepare_stack

#%% Inputs --------------------------------------------------------------------

rf = 0.5
nSlice = 5

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))   
train_path = Path("data", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in stk_paths:
    
        # Format and merge stack
        stk, voxsize = format_stack(path, rf=rf)
        prp = prepare_stack(stk, voxsize)
        
        # Save random slices
        idxs = np.random.choice(
            np.arange(prp.shape[0]), size=nSlice, replace=False)
        for idx in idxs:
            io.imsave(
                train_path / (path.stem + f"_{voxsize}_z{idx:02d}_mrg.tif"),
                prp[idx, ...], check_contrast=False
                )