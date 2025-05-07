#%% Imports -------------------------------------------------------------------

import nd2
import time
import pickle
import shutil
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# skimage
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.exposure import adjust_gamma
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import (
    ball, h_maxima, remove_small_objects, remove_small_holes)

#%% Inputs --------------------------------------------------------------------

procedure = {
    "extract" : 0,
    "predict" : 0,
    "process" : 0,
    "display" : 1,
    }

parameters = {
    
    # Paths
    "data_path"   : Path("D:\local_Suzuki\data"),
    "model_name"  : "model_512_normal_2000-160_3",
    "tags"        : ["2OBJ"],
    
    # Parameters
    "voxsize"     : 0.2,
    "cyt_thresh"  : 0.05,
    "ncl_thresh"  : 0.15,
    "blb_threshs" : {
        "2obj"    : [0.75, 0.33, 0.25], # C1, C2, C3
        "3obj"    : [0.75, 0.33, 0.25], # C1, C2, C3
        },
    
    # Channels
    "chn_names"   : {
        "2obj"    : ["NRP2", "virus-all", "virus-extra", "nucleus"],
        "3obj"    : ["NRP2", "virus-all", "EEA1", "nucleus"],
        },
    
    # Display
    "cmaps"       : {
        "2obj"    : ["bop orange", "magenta", "green", "blue"],
        "3obj"    : ["bop orange", "magenta", "green", "blue"],
        },
    
    } 

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
    "Dr01" : "DMSO",
    "Dr02" : "Dyngo",
    "Dr03" : "EIPA",
    "Dr05" : "CPZ",
    
    }

#%% Function(s) ---------------------------------------------------------------

def save_tif(arr, path, voxsize):
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
        data["cyt_msk"] = io.imread(msk_path)
        data["ncl_msk"] = io.imread(out_path / "ncl_msk.tif")
    
    # Load blb
    blb_path = out_path / "C1_blb.tif"
    if blb_path.exists():
        data["C1_blb"] = io.imread(blb_path)
        data["C2_blb"] = io.imread(out_path / "C2_blb.tif")
        data["C3_blb"] = io.imread(out_path / "C3_blb.tif")
        
    return data

#%% Class(Main) ---------------------------------------------------------------

class Main:
    
    def __init__(
            self, 
            procedure=procedure, 
            parameters=parameters,
            mapping=mapping,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.mapping    = mapping
        
        # Initialize
        if "2OBJ" in self.parameters["tags"]:
            self.exp = "2obj"
        if "3OBJ" in self.parameters["tags"]:
            self.exp = "3obj"
        
        # Run
        self.initialize()
        if self.procedure["extract"]:
            self.extract() 
        if self.procedure["predict"]:
            self.predict() 
        if self.procedure["process"]:
            self.process() 
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        # Paths
        paths = list(self.parameters["data_path"].rglob("*.nd2"))
        self.htk_paths = []
        for path in paths:
            if any(tag in path.stem for tag in self.parameters["tags"]):
                self.htk_paths.append(path)
                    
#%% Class(Main) : extract() ---------------------------------------------------
        
    def extract(self):
        
        def load_rescale(path):
            
            with nd2.ND2File(path) as ndfile:
                
                # voxsize
                voxsize = (
                    ndfile.voxel_size()[2],
                    ndfile.voxel_size()[1],
                    ndfile.voxel_size()[0],
                    )
                
                # Determine rescaling factors (rfi & rfc)
                rfi = voxsize[1] / voxsize[0]
                rfc = voxsize[0] / self.parameters["voxsize"] 
                
                # Load & rescale hstack
                htk = ndfile.asarray()
                shape0 = htk.shape
                htk = rescale(htk, (1, 1, rfi, rfi),   order=0) # iso
                htk = rescale(htk, (rfc, 1, rfc, rfc), order=0) # custom
                shape1 = htk.shape    
                
                # Flip z axis
                htk = np.flip(htk, axis=0)
                
                # Metadata
                metadata = {
                    "path"     : path,
                    "shape0"   : shape0,
                    "shape1"   : shape1,
                    "voxsize0" : voxsize,
                    "voxsize1" : (self.parameters["voxsize"],) * 3,
                    }
                                    
            return metadata, htk
            
        def _extract(path):
            
            # Setup directory
            out_path = path.parent / path.stem        
            if out_path.exists():
                for item in out_path.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                    
            # Load & rescale
            metadata, htk = load_rescale(path) 
            
            # Save
            
            with open(str(out_path / "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            for c in range(htk.shape[1]):
                stk = htk[:, c, ...]
                stk = norm_pct(stk, sample_fraction=0.1)
                stk = (stk * 255).astype("uint8")
                save_tif(
                    stk, out_path / f"C{c + 1}.tif", 
                    metadata["voxsize1"][0]
                    )
                
        # ---------------------------------------------------------------------
        
        t0 = time.time()
        print("extract() : ", end="", flush=False)
        
        Parallel(n_jobs=-1)(
            delayed(_extract)(path) 
            for path in self.htk_paths
            ) 
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : predict() ---------------------------------------------------
    
    def predict(self):
        
        def prepare_stack(C1, C4):
            C1 = adjust_gamma(norm_pct(C1), gamma=0.5)
            C4 = adjust_gamma(norm_pct(C4), gamma=0.5)
            return (C1 + C4 ) / 3
        
        # ---------------------------------------------------------------------
    
        t0 = time.time()
        print("predict() : ", end="", flush=False)
        
        # Initialize model
        unet = UNet(load_name=self.parameters["model_name"])
        
        for path in self.htk_paths:
                        
            # Load data
            out_path = path.parent / path.stem
            data = load_data(out_path)
            
            # Prepare stack
            prp = prepare_stack(
                data["htk"][:, 0, ...], 
                data["htk"][:, 3, ...],
                )
            
            # Predict
            prd = unet.predict(prp, verbose=0)
            
            # Save
            prd = (prd * 255).astype("uint8")
            save_tif(
                prd, out_path / "prd.tif", 
                data["metadata"]["voxsize1"][0],
                )         
            
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : process() ---------------------------------------------------

    def process(self):
        
        def get_masks(htk, prd):
            
            # Process
            prd_prp = norm_pct(gaussian(prd, sigma=2))
            cyt_prp = norm_pct(gaussian(htk[:, 0, ...], sigma=1))
            ncl_prp = norm_pct(gaussian(htk[:, 3, ...], sigma=2))
            cyt_prp *= prd_prp
            ncl_prp *= prd_prp
            cyt_msk = cyt_prp > self.parameters["cyt_thresh"]
            ncl_msk = ncl_prp > self.parameters["ncl_thresh"]
            cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
            ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
            
            # Posprocess ncl_msk
            for z in range(ncl_msk.shape[0]):
                ncl_msk[z, ...] = remove_small_holes(ncl_msk[z, ...])
            ncl_lbl = label(ncl_msk)
            for props in regionprops(ncl_lbl, intensity_image=cyt_msk):
                lbl = props.label
                val = props.intensity_mean
                if val < 0.25:
                    ncl_lbl[ncl_lbl == lbl] = 0
            ncl_msk = ncl_lbl > 0
            
            # Postprocess cyt_msk
            cyt_msk[ncl_msk] = 1
            
            return cyt_msk, ncl_msk
        
        def get_blobs(
                arr, mask=None, 
                sigma0=0.5, sigma1=5, thresh=0.5, 
                out_prp=False
                ):
            
            # preprocess
            prp = gaussian(arr, sigma=sigma0)
            prp -= gaussian(prp, sigma=sigma1)
            prp = norm_pct(prp)
            if mask is not None:
                prp[mask == 0] = 0
            msk = prp > thresh
            
            # watershed
            hmax = h_maxima(prp, 0.1, footprint=ball(1))
            hmax = (hmax * 255).astype("uint8")
            hmax = label(hmax)
            lbl = watershed(-prp, markers=hmax, mask=msk)
            
            if out_prp:
                return lbl, prp
            else:
                return lbl
            
        def _process(path):
            
            # Load data
            out_path = path.parent / path.stem
            data = load_data(out_path)
            
            # Get masks
            cyt_msk, ncl_msk = get_masks(data["htk"], data["prd"])
            
            for c in range(3):
                
                # Get blobs
                blb_lbl = get_blobs(
                    data["htk"][:, c, ...], mask=cyt_msk, 
                    sigma0=0.5, sigma1=5, 
                    thresh=self.parameters["blb_threshs"][self.exp][c]
                    )
                        
                # Save
                save_tif(
                    (cyt_msk * 255).astype("uint8"), 
                    out_path / "cyt_msk.tif", 
                    data["metadata"]["voxsize1"][0],
                    )   
                save_tif(
                    (ncl_msk * 255).astype("uint8"), 
                    out_path / "ncl_msk.tif", 
                    data["metadata"]["voxsize1"][0],
                    )   
                save_tif(
                    blb_lbl.astype("uint16"), 
                    out_path / f"C{c + 1}_blb.tif", 
                    data["metadata"]["voxsize1"][0],
                    )   

        # ---------------------------------------------------------------------
        
        t0 = time.time()
        print("process() : ", end="", flush=False)
        
        Parallel(n_jobs=-1)(
            delayed(_process)(path) 
            for path in self.htk_paths
            ) 

        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Display) ------------------------------------------------------------
    
class Display:
    
    def __init__(
            self, 
            procedure=procedure, 
            parameters=parameters,
            mapping=mapping,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.mapping    = mapping
        
        # Initialize
        self.idx = 0
        if "2OBJ" in self.parameters["tags"]:
            self.exp = "2obj"
        if "3OBJ" in self.parameters["tags"]:
            self.exp = "3obj"
        
        # Run
        if self.procedure["display"]:
            self.init_data()
            self.init_viewer()
        
#%% Class(Display) : init_data() ----------------------------------------------
        
    def init_data(self):
        
        # Paths
        paths = list(self.parameters["data_path"].rglob("*.nd2"))
        self.htk_paths = []
        for path in paths:
            if any(tag in path.stem for tag in self.parameters["tags"]):
                self.htk_paths.append(path)
                
        # Parameters
        self.chn_names = self.parameters["chn_names"][self.exp]
        self.cmaps = self.parameters["cmaps"][self.exp]
        
        # Load & format data
        self.data = []
        for path in self.htk_paths:
            out_path = path.parent / path.stem
            data = load_data(out_path)
            self.data.append(data)
                
#%% Class(Display) : init_viewer() --------------------------------------------                

    def init_viewer(self):
                
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Add images
        for c in range(self.data[0]["htk"].shape[1]):
            
            # raws
            self.viewer.add_image(
                self.data[0]["htk"][:, c, ...], visible=0,
                name=self.chn_names[c], colormap=self.cmaps[c], 
                blending="additive", gamma=0.75, 
                )
            
            if c < 3:

                # blobs
                self.viewer.add_image(
                    self.data[0][f"C{c + 1}_blb"] > 0, visible=1,
                    name=self.chn_names[c] + "_blb", colormap=self.cmaps[c],
                    blending="additive", opacity=0.75, 
                    rendering="attenuated_mip", attenuation=0.5,  
                    )
                
        # masks
        self.viewer.add_image(
            self.data[0]["cyt_msk"], visible=0,
            name="cyt_msk", colormap="gray",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        self.viewer.add_image(
            self.data[0]["ncl_msk"], visible=0,
            name="ncl_msk", colormap="blue",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main()
    Display()
    