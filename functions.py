#%% Imports -------------------------------------------------------------------

import nd2
import pickle
import tifffile
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.transform import rescale
from skimage.exposure import adjust_gamma

#%% Mapping -------------------------------------------------------------------

mapping = {

    # NRP2
    "E00"  : "eGFP",
    "N00"  : "NRP2-eGFP",
    "N01"  : "NRP2-eGFP_T319R",
    "N02"  : "NRP2-eGFP_AAA",
    "N03"  : "NRP2-eGFP_dA1A2",
    "N04"  : "NRP2-eGFP_dB1",
    "N05"  : "NRP2-eGFP_dB2",
    "N06"  : "NRP2-eGFP_dMAM",
    "N07"  : "NRP2-eGFP_dCyto",
    "N08"  : "NRP2-eGFP_noSA",
    
    # Drugs
    "Im00" : "none",
    "IM00" : "none",
    "Dr01" : "DMSO",
    "Dr02" : "Dyngo",
    "Dr03" : "EIPA",
    "Dr05" : "CPZ",
    
    # Channels
    "chn_names" : {
        "2obj"  : ["_", "virus-all", "virus-extra", "nucleus"],
        "3obj"  : ["_", "virus-all", "EEA1", "nucleus"],
        },
    
    }

#%% Function : import_htk() ---------------------------------------------------

def import_htk(path, voxsize=0.2):
    
    if "2OBJ" in path.name: exp = "2obj"
    if "3OBJ" in path.name: exp = "3obj"
                
    with nd2.ND2File(path) as ndfile:
        
        # vsize_in
        vsize0 = (
            ndfile.voxel_size()[2],
            ndfile.voxel_size()[1],
            ndfile.voxel_size()[0],
            )
        
        # Load 
        htk = ndfile.asarray()
        
    # Determine rescaling factors (rfi & rfc)
    rfi = vsize0[1] / vsize0[0]
    rfc = vsize0[0] / voxsize
        
    # Load & rescale hstack
    shape0 = htk.shape
    htk = rescale(htk, (1, 1, rfi, rfi),   order=0) # iso
    htk = rescale(htk, (rfc, 1, rfc, rfc), order=0) # custom
    shape1 = htk.shape    
    
    # Flip z axis
    htk = np.flip(htk, axis=0)
    
    # Metadata
    cond = path.stem.split("_")[5]
    chn1 = path.stem.split("_")[4]
    chn_names = mapping["chn_names"][exp]
    chn_names[0] = mapping[chn1]
    metadata = {
        "path"      : path,
        "cond"      : mapping[cond],
        "chn_names" : chn_names,
        "shape0"    : shape0,
        "shape1"    : shape1,
        "vsize0"    : vsize0,
        "vsize1"    : (voxsize,) * 3,
        }
                            
    return metadata, htk

#%% Function : prepare_htk() --------------------------------------------------

def prepare_htk(htk):
    C1, C4 = htk[:, 0, ...], htk[:, 3, ...]
    C1 = adjust_gamma(norm_pct(C1), gamma=0.5)
    C4 = adjust_gamma(norm_pct(C4), gamma=0.5)
    return (C1 + C4) / 3

#%% Function : save_tif() -----------------------------------------------------
    
def save_tif(arr, path, voxsize=0.2):
    if arr.ndim == 2: axes = "YX"
    if arr.ndim == 3: axes = "ZYX"
    if arr.ndim == 4: axes = "ZCYX"
    resolution = (1 / voxsize, 1 / voxsize)
    metadata = {"axes" : axes, "spacing" : voxsize, "unit" : "um"}
    tifffile.imwrite(
        path, arr, 
        imagej=True, 
        resolution=resolution, 
        metadata=metadata
        )

#%% Function : load_data() ----------------------------------------------------

def load_data(out_path):
    
    data = {}
    
    # Load metadata & htk
    with open(str(out_path / "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    htk = np.zeros(metadata["shape1"], dtype="uint16")
    for c in range(htk.shape[1]):
        htk[:, c, ...] = io.imread(out_path / f"C{c + 1}.tif")    
    data["metadata"] = metadata
    data["htk"] = htk
    
    # Load prd
    prd_path = out_path / "prd.tif"
    if prd_path.exists():
        data["prd"] = io.imread(out_path / "prd.tif")
       
    # Load msk
    msk_path = out_path / "cyt_msk.tif"
    if msk_path.exists():   
        data["cyt_msk"] = io.imread(out_path / "cyt_msk.tif")
        data["ncl_msk"] = io.imread(out_path / "ncl_msk.tif")
    
    # Load blb
    blb_path = out_path / "C1_lbl.tif"
    if blb_path.exists():
        for c in range(3):
            data[f"C{c + 1}_lbl"] = io.imread(out_path / f"C{c + 1}_lbl.tif")
    
    # Load blb_f
    blb_f_path = out_path / "C2_lbl_f.tif"
    if blb_f_path.exists():
        data["C2_lbl_f"] = io.imread(out_path / "C2_lbl_f.tif")
        
    return data

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    idx = 0
    data_path = Path("D:\local_Suzuki\data\\2obj")
    htk_paths = list(data_path.rglob("*.nd2"))
    
    # Load
    metadata, htk = import_htk(htk_paths[idx], vsize1=0.2)
    prp = prepare_htk(htk)
    
    # Display
    import napari
    viewer = napari.Viewer()
    viewer.add_image(prp)