#%% Imports -------------------------------------------------------------------

import nd2
import time
import shutil
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

procedure = {
    "extract" : 1,
    }

parameters = {
    "data_path"  : Path("D:\local_Suzuki\data"),
    "model_name" : "model_512_normal_2000-160_3",
    "tags"       : ["2OBJ"],
    "voxsize"    : 0.2,
    } 

#%%

def save(arr, path, voxsize):
    if arr.ndim == 3: axes = "ZYX"
    if arr.ndim == 4: axes = "ZCYX"
    resolution = (1 / voxsize, 1 / voxsize)
    metadata = {"axes" : axes, "spacing" : voxsize, "unit" : "um"}
    tifffile.imwrite(
        path, arr, imagej=True, resolution=resolution, metadata=metadata)

#%% Class : Main --------------------------------------------------------------

class Main:
    
    def __init__(self, procedure=procedure, parameters=parameters):
        
        # Fetch
        self.procedure = procedure
        self.data_path = parameters["data_path"]
        self.model_name = parameters["model_name"]
        self.tags = parameters["tags"]
        self.voxsize = parameters["voxsize"]
        
        # Run
        self.initialize()
        if self.procedure["extract"]:
            self.extract() 
        
#%% Method : initialize() -----------------------------------------------------

    def initialize(self):
        
        # Paths
        paths = list(self.data_path.rglob("*.nd2"))
        self.htk_paths = []
        for path in paths:
            if any(tag in path.stem for tag in self.tags):
                self.htk_paths.append(path)
                
#%% Method : load() -----------------------------------------------------------

    def load(self, path):
        
        with nd2.ND2File(path) as ndfile:
            
            # voxsize
            voxsize = (
                ndfile.voxel_size()[2],
                ndfile.voxel_size()[1],
                ndfile.voxel_size()[0],
                )
            
            # Determine rescaling factors (rfi & rfc)
            rfi = voxsize[1] / voxsize[0]
            rfc = voxsize[0] / self.voxsize 
            
            # Load & rescale hstack
            htk = ndfile.asarray()
            htk = rescale(htk, (1, 1, rfi, rfi),   order=0) # iso
            htk = rescale(htk, (rfc, 1, rfc, rfc), order=0) # custom
                
            # Flip z axis
            htk = np.flip(htk, axis=0)
            
            # Metadata
            metadata = {
                "path"     : path,
                "voxsize0" : voxsize,
                "voxsize1" : (rfc, rfc, rfc),
                }
                                
        return metadata, htk
    
#%% Method : extract() --------------------------------------------------------
        
    def extract(self):
        
        def _extract(path):
            
            # Setup directory
            save_path = path.parent / path.stem        
            if save_path.exists():
                for item in save_path.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                save_path.mkdir(parents=True, exist_ok=True)
                    
            # Load
            metadata, htk = self.load(path) 
            
            # Save
            for c in range(htk.shape[1]):
                io.imsave(
                    save_path / f"C{c}.tif",
                    htk[:, c, ...], check_contrast=False,
                    )
        
        t0 = time.time()
        print("extract() : ", end="", flush=False)
        
        Parallel(n_jobs=-1)(
            delayed(_extract)(path) 
            for path in self.htk_paths
            ) 
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # path = self.htk_paths[0]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    main = Main()
    
#%%
       
    # # Display
    # viewer = napari.Viewer()
    # for c in range(htk.shape[1]):
    #     viewer.add_image(
    #         htk[:, c, ...], name=f"C{c + 1}",
    #         )
    
    