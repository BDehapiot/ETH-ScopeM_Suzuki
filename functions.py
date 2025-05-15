#%% Imports -------------------------------------------------------------------

import nd2
import pickle
import warnings   
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

# def import_htk(path, voxsize=0.2):
    
#     if "2OBJ" in path.name: exp = "2obj"
#     if "3OBJ" in path.name: exp = "3obj"
                
#     with nd2.ND2File(path) as ndfile:
        
#         # vsize_in
#         vsize0 = (
#             ndfile.voxel_size()[2],
#             ndfile.voxel_size()[1],
#             ndfile.voxel_size()[0],
#             )
        
#         # Load 
#         htk = ndfile.asarray()
        
#     # Determine rescaling factors (rfi & rfc)
#     rfi = vsize0[1] / vsize0[0]
#     rfc = vsize0[0] / voxsize
        
#     # Load & rescale hstack
#     shape0 = htk.shape
#     htk = rescale(htk, (  1, 1, rfi, rfi), order=0) # iso
#     htk = rescale(htk, (rfc, 1, rfc, rfc), order=0) # custom
#     shape1 = htk.shape    
    
#     # Convert to "uint8" (from 0-4095 to 0-255)
#     htk = (htk // 16).astype("uint8")
    
#     # Flip z axis
#     htk = np.flip(htk, axis=0)
    
#     # Metadata
#     cond = path.stem.split("_")[5]
#     chn1 = path.stem.split("_")[4]
#     chn_names = mapping["chn_names"][exp]
#     chn_names[0] = mapping[chn1]
#     metadata = {
#         "path"      : path,
#         "cond"      : mapping[cond],
#         "chn_names" : chn_names,
#         "shape0"    : shape0,
#         "shape1"    : shape1,
#         "vsize0"    : vsize0,
#         "vsize1"    : (voxsize,) * 3,
#         }
                            
#     return metadata, htk

#%% Function : import_nd2() ---------------------------------------------------

def check_nd2(path):
    
    # Initialize
    warnings.filterwarnings(
        "ignore", 
        message="ND2File file not closed before garbage collection"
        )
    
    with nd2.ND2File(path) as f:
        darr = f.to_dask()
    
    return darr.shape

def import_nd2(path, z="all", c="all", voxsize=0.2):
    
    # Initialize
    if "2OBJ" in path.name: exp = "2obj"
    if "3OBJ" in path.name: exp = "3obj"
    zi = slice(None) if z == "all" else z 
    ci = slice(None) if c == "all" else c 
    warnings.filterwarnings(
        "ignore", 
        message="ND2File file not closed before garbage collection"
        )
                    
    with nd2.ND2File(path) as f:
        
        # Input voxel size (vsize0)
        vsize0 = (
            f.voxel_size()[2],
            f.voxel_size()[1],
            f.voxel_size()[0],
            )
        
        # Load        
        darr = f.to_dask()
        arr  = darr[zi, ci, ...].compute()
    
    # Determine rescaling factors (rfi & rfc)
    rfi = vsize0[1] / vsize0[0]
    rfc = vsize0[0] / voxsize
    if arr.ndim == 4:
        rscale0 = (1, 1, rfi, rfi)
        rscale1 = (rfc, 1, rfc, rfc)
    if arr.ndim == 3 and z == "all":
        rscale0 = (1, rfi, rfi)
        rscale1 = (rfc, rfc, rfc)
    if arr.ndim == 3 and c == "all":
        rscale0 = (1, rfi, rfi)
        rscale1 = (1, rfc, rfc)
    if arr.ndim == 2:
        rscale0 = (rfi, rfi)
        rscale1 = (rfc, rfc)
    
    # Rescale array
    shape0 = arr.shape
    arr = rescale(arr, rscale0, order=0) # iso
    arr = rescale(arr, rscale1, order=0) # custom
    shape1 = arr.shape   
    
    # Convert to "uint8" (from 0-4095 to 0-255)
    arr = (arr // 16)
    arr = np.clip(arr, 0, 255)
    arr = arr.astype("uint8")
    
    # Flip z axis
    # if arr.ndim == 4:
    #     arr = np.flip(arr, axis=0)
    
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
                            
    return metadata, arr

#%% Function : prepare_htk() --------------------------------------------------

# def prepare_htk(htk):
#     C1, C4 = htk[:, 0, ...], htk[:, 3, ...]
#     mrg = (C1 + C4) / 2
#     max0 = np.max(mrg)
#     mrg = adjust_gamma(mrg, gamma=0.5)
#     max1 = np.max(mrg)
#     mrg *= max0 / max1 
#     return mrg.astype("uint8")

def prepare_data(C1, C4):
    mrg = np.maximum(C1, C4)
    return mrg

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
    
    import time

    data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Suzuki\data")
    nd2_paths = list(data_path.rglob("*.nd2"))  
    
    # -------------------------------------------------------------------------
    
    name = "20250317_rep2_cell07_000min_N00_Dr01_2OBJ_none_.nd2"
    path = list(data_path.rglob(f"*{name}"))[0]
    voxsize = 0.2
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("open : ", end="", flush=False)
    
    shape = check_nd2(path)
    if shape[1] == 4:
        _, C1 = import_nd2(path, z=13, c=0, voxsize=voxsize)
        _, C4 = import_nd2(path, z=13, c=3, voxsize=voxsize)
        prp = prepare_data(C1, C4)
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    import napari
    viewer = napari.Viewer()
    viewer.add_image(prp)