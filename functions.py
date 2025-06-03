#%% Imports -------------------------------------------------------------------

import nd2
import pickle
import warnings   
import tifffile
import numpy as np
from skimage import io
from pathlib import Path

# skimage
from skimage.transform import rescale
from skimage.exposure import adjust_gamma

#%% Mapping -------------------------------------------------------------------

mapping = {

    # NRP2
    "E00"  : "NRP2KO-eGFP",
    "N00"  : "NRP2KO-NRP2-eGFP",
    "N01"  : "NRP2KO-NRP2-eGFP_T319R",
    "N02"  : "NRP2KO-NRP2-eGFP_AAA",
    "N03"  : "NRP2KO-NRP2-eGFP_dA1A2",
    "N04"  : "NRP2KO-NRP2-eGFP_dB1",
    "N05"  : "NRP2KO-NRP2-eGFP_dB2",
    "N06"  : "NRP2KO-NRP2-eGFP_dMAM",
    "N07"  : "NRP2KO-NRP2-eGFP_dCyto",
    "N08"  : "NRP2KO-NRP2-eGFP_dSA",
    "N09"  : "NRP2KO-NRP2-eGFP_dA1A2B1B2",
    "N10"  : "NRP2KO-NRP2-eGFP_dB1B2",
    "N11"  : "NRP2KO-NRP2-eGFP_dSAB1",
    "N12"  : "NRP2KO-NRP2-eGFP_dSAB1B2",
    
    # Drugs
    "IM00" : "none",
    "Dr01" : "DMSO",
    "Dr02" : "Dyngo4a",
    "Dr03" : "EIPA",
    "Dr04" : "Pitstop2",
    "Dr05" : "CPZ",
    
    # Channels
    "chn_names" : {
        "2obj"  : ["_", "virus-all", "virus-extra", "nucleus"],
        "3obj"  : ["_", "virus-all", "EEA1", "nucleus"],
        },
    
    }

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
    date = path.stem.split("_")[0]
    rep  = int(path.stem.split("_")[1][-1])
    cell = int(path.stem.split("_")[2][-2:])
    time = int(path.stem.split("_")[3][:3])
    chn1 = path.stem.split("_")[4]
    cond = path.stem.split("_")[5]
    chn_names = mapping["chn_names"][exp]
    chn_names[0] = mapping[chn1]
    metadata = {
        "path"      : path,
        "date"      : date,
        "rep"       : rep,
        "cell"      : cell,
        "time"      : time,
        "chn_names" : chn_names,
        "cond"      : mapping[cond],
        "exp"       : exp,
        "shape0"    : shape0,
        "shape1"    : shape1,
        "vsize0"    : vsize0,
        "vsize1"    : (voxsize,) * 3,
        }
                            
    return metadata, arr

#%% Function : prepare_htk() --------------------------------------------------

def prepare_data(C1, C4):
    prp = np.maximum(C1, C4)
    prp = adjust_gamma(prp, gamma=0.33)
    return prp

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
    cyt_prd_path = out_path / "cyt_prd.tif"
    if cyt_prd_path.exists():
        data["cyt_prd"] = io.imread(out_path / "cyt_prd.tif")
        data["ncl_prd"] = io.imread(out_path / "ncl_prd.tif")
       
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
    
    # Load blb_v
    blb_v_path = out_path / "C2_lbl_v.tif"
    if blb_v_path.exists():
        data["C2_lbl_v"] = io.imread(out_path / "C2_lbl_v.tif")
        
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
        metadata, C1 = import_nd2(path, z=13, c=0, voxsize=voxsize)
        metadata, C4 = import_nd2(path, z=13, c=3, voxsize=voxsize)
        prp = prepare_data(C1, C4)
                
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    
    # -------------------------------------------------------------------------
    
    import napari
    viewer = napari.Viewer()
    viewer.add_image(prp)