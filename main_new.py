#%% Imports -------------------------------------------------------------------

import nd2
import time
import pickle
import shutil
import napari
import tifffile
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# skimage
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.exposure import adjust_gamma
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import (
    ball, h_maxima, binary_dilation, remove_small_objects, remove_small_holes)

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

# matplotlib
import matplotlib.pyplot as plt 

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
        "2obj"    : ["_", "virus-all", "virus-extra", "nucleus"],
        "3obj"    : ["_", "virus-all", "EEA1", "nucleus"],
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
    "Im00" : "none",
    "IM00" : "none",
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
    blb_path = out_path / "C1_lbl.tif"
    if blb_path.exists():
        for c in range(3):
            data[f"C{c + 1}_lbl"] = io.imread(out_path / f"C{c + 1}_lbl.tif")
        
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
        if "2OBJ" in self.parameters["tags"]:
            self.exp = "2obj"
        if "3OBJ" in self.parameters["tags"]:
            self.exp = "3obj"
                    
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
                cond = path.stem.split("_")[5]
                chn1 = path.stem.split("_")[4]
                chn_names = self.parameters["chn_names"][self.exp]
                chn_names[0] = self.mapping[chn1]
                metadata = {
                    "path"      : path,
                    "cond"      : self.mapping[cond],
                    "chn_names" : chn_names,
                    "shape0"    : shape0,
                    "shape1"    : shape1,
                    "voxsize0"  : voxsize,
                    "voxsize1"  : (self.parameters["voxsize"],) * 3,
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
                
            metadata_df = pd.DataFrame(
                list(metadata.items()), columns=['Key', 'Value'])
            metadata_df.to_csv(out_path / "metadata.csv", index=False)
            
            for c in range(htk.shape[1]):
                stk = htk[:, c, ...]
                stk = (stk // 16).astype("uint8") 
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
                    out_path / f"C{c + 1}_lbl.tif", 
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
        
        def get_outlines(arr):
            arr = arr > 0
            out = []
            for img in arr:
                out.append(binary_dilation(img) ^ img)
            return np.stack(out)
        
        # Paths
        paths = list(self.parameters["data_path"].rglob("*.nd2"))
        self.htk_paths = []
        for path in paths:
            if any(tag in path.stem for tag in self.parameters["tags"]):
                self.htk_paths.append(path)
                
        # Parameters
        self.cmaps = self.parameters["cmaps"][self.exp]
        
        # Load & format data
        self.data = []
        for path in self.htk_paths:
            out_path = path.parent / path.stem
            data = load_data(out_path)
            data["cyt_out"] = get_outlines(data["cyt_msk"])
            data["ncl_out"] = get_outlines(data["ncl_msk"])
            for c in range(3):
                data[f"C{c + 1}_out"] = get_outlines(data[f"C{c + 1}_lbl"] > 0)
            self.data.append(data)
                
#%% Class(Display) : init_viewer() --------------------------------------------                

    def init_viewer(self):
                
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Create "hstack" menu
        self.htk_group_box = QGroupBox("Select hstack")
        htk_group_layout = QVBoxLayout()
        self.btn_next_htk = QPushButton("next")
        self.btn_prev_htk = QPushButton("prev")
        htk_group_layout.addWidget(self.btn_next_htk)
        htk_group_layout.addWidget(self.btn_prev_htk)
        self.htk_group_box.setLayout(htk_group_layout)
        self.btn_next_htk.clicked.connect(self.next_hstack)
        self.btn_prev_htk.clicked.connect(self.prev_hstack)
                
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QHBoxLayout()
        self.rad_masks = QRadioButton("mask")
        self.rad_predictions = QRadioButton("predictions")
        self.rad_masks.setChecked(True)
        dsp_group_layout.addWidget(self.rad_masks)
        dsp_group_layout.addWidget(self.rad_predictions)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_masks.toggled.connect(
            lambda checked: self.show_masks() if checked else None)
        self.rad_predictions.toggled.connect(
            lambda checked: self.show_predictions() if checked else None)
        
        # Create "check" menu
        self.chk_group_box = QGroupBox("Check blobs")
        chk_group_layout = QHBoxLayout()
        self.rad_chk_C1b = QRadioButton("C1")
        self.rad_chk_C2b = QRadioButton("C2")
        self.rad_chk_C3b = QRadioButton("C3")
        chk_group_layout.addWidget(self.rad_chk_C1b)
        chk_group_layout.addWidget(self.rad_chk_C2b)
        chk_group_layout.addWidget(self.rad_chk_C3b)
        self.chk_group_box.setLayout(chk_group_layout)
        self.rad_chk_C1b.toggled.connect(
            lambda checked: self.show_chk(tag="C1") if checked else None)
        self.rad_chk_C2b.toggled.connect(
            lambda checked: self.show_chk(tag="C2") if checked else None)
        self.rad_chk_C3b.toggled.connect(
            lambda checked: self.show_chk(tag="C3") if checked else None)
        
        # Create texts
        self.info_path = QLabel()
        self.info_path.setFont(QFont("Consolas"))
        self.info_path.setText(self.get_text())

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.htk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addWidget(self.chk_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_path)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        self.init_layers()        
        
        # Shortcuts
        
        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_hstack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_hstack()
                                
    def next_hstack(self):
        if self.idx < len(self.data) - 1:
            self.idx += 1
            self.update_layers()
            self.update_text()
            
    def prev_hstack(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_layers()
            self.update_text()
                    
    def center_view(self):
        for name in self.viewer.layers:
            name = str(name)
            shape = self.viewer.layers[name].data.shape
            self.viewer.camera.center = (
                shape[0] // 2, shape[1] // 2, shape[2] // 2)
            self.viewer.camera.zoom = 1.0
            
    def show_masks(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if "msk" in name:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
    
    def show_predictions(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C1", "prd"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
                
    def show_chk(self, tag="C1"):
        self.viewer.dims.ndisplay = 2
        for name in self.viewer.layers:
            name = str(name)
            if name in [f"{tag}", f"{tag}_out", "cyt_out", "ncl_out"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
                
#%% Class(Display) : init_layers() --------------------------------------------            

    def init_layers(self):
        
        for c in range(self.data[0]["htk"].shape[1]- 1, -1, -1):
            
            if c < 3:

                # blobs
                self.viewer.add_image(
                    self.data[0][f"C{c + 1}_lbl"] > 0, visible=1,
                    name=f"C{c + 1}_msk", colormap=self.cmaps[c],
                    blending="additive", opacity=0.75, 
                    rendering="attenuated_mip", attenuation=0.5,  
                    )
                self.viewer.add_image(
                    self.data[0][f"C{c + 1}_out"], visible=0,
                    name=f"C{c + 1}_out", colormap="gray",
                    blending="additive", opacity=0.5,  
                    ) 
                
            # htk
            self.viewer.add_image(
                self.data[0]["htk"][:, c, ...], visible=0,
                name=f"C{c + 1}", colormap=self.cmaps[c], 
                blending="additive", gamma=0.5, 
                )
                            
        # masks
        
        self.viewer.add_image(
            self.data[0]["ncl_msk"], visible=1,
            name="ncl_msk", colormap="blue",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.data[0]["ncl_out"], visible=0,
            name="ncl_out", colormap="gray",
            blending="additive", opacity=0.2,  
            )
        
        self.viewer.add_image(
            self.data[0]["cyt_msk"], visible=1,
            name="cyt_msk", colormap="gray",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.data[0]["cyt_out"], visible=0,
            name="cyt_out", colormap="gray",
            blending="additive", opacity=0.2,  
            )
        
        # prediction
        self.viewer.add_image(
            self.data[0]["prd"], visible=0,
            name="prd", colormap="turbo",
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        # Adjust viewer
        self.viewer.dims.ndisplay = 3
        self.center_view()
        
#%% Class(Display) : update() -------------------------------------------------  

    def update_layers(self):
        
        for c in range(self.data[self.idx]["htk"].shape[1]):
            
            if c < 3: 
                
                # blobs
                self.viewer.layers[f"C{c + 1}_msk"].data = (
                    self.data[self.idx][f"C{c + 1}_lbl"] > 0)
                self.viewer.layers[f"C{c + 1}_out"].data = (
                    self.data[self.idx][f"C{c + 1}_out"] > 0)
            
            # htk
            self.viewer.layers[f"C{c + 1}"].data = (
                self.data[self.idx]["htk"][:, c, ...])

        # masks
        self.viewer.layers["ncl_msk"].data = self.data[self.idx]["ncl_msk"]
        self.viewer.layers["cyt_msk"].data = self.data[self.idx]["cyt_msk"]
        self.viewer.layers["ncl_out"].data = self.data[self.idx]["ncl_out"]
        self.viewer.layers["cyt_out"].data = self.data[self.idx]["cyt_out"]
        
        # prediction
        self.viewer.layers["prd"].data = self.data[self.idx]["prd"]
        
        # Adjust viewer
        self.center_view()
        
    def update_text(self):
        self.info_path.setText(self.get_text())

#%% Class(Display) : get_text() -----------------------------------------------

    def get_text(self):
        
        def format_tuple(tpl, fmt=".3f", sep=", "):
            fmt_tpl = []
            for elm in tpl:
                fmt_tpl.append(f"{elm:{fmt}}")
            return sep.join(fmt_tpl)
        
        # ---------------------------------------------------------------------
        
        metadata = self.data[self.idx]['metadata']
        
        # Formatting
        chn_names = [
            f"C{c + 1}   : {metadata['chn_names'][c]}\n"
            for c in range(self.data[self.idx]["htk"].shape[1])
            ]
        shape0 = format_tuple(metadata["shape0"], fmt="04d")
        shape1 = format_tuple(metadata["shape1"], fmt="04d")
        voxsize0 = format_tuple(metadata["voxsize0"], fmt=".3f")
        voxsize1 = format_tuple(metadata["voxsize1"], fmt=".3f")
        
        return (
            
            f"{self.data[self.idx]['metadata']['path'].name}"
            
            "\n\n"
            f"cond : {metadata['cond']}" 
            "\n"
            f"{''.join(chn_names)}"
            
            "\n"
            f"shape0 = ({shape0})"
            "\n"
            f"shape1 = ({shape1})"
            "\n"
            f"voxsize0 = ({voxsize0})"
            "\n"
            f"voxsize1 = ({voxsize1})"
            "\n"
   
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    main = Main()
    display = Display()
    
#%%

    idx = 5

    def get_obj_int(lbl, img):
        lbl_f = lbl.ravel()
        img_f = img.ravel()        
        counts = np.bincount(lbl_f)
        sum_img = np.bincount(lbl_f, weights=img_f)
        idx = np.nonzero(counts)[0]
        idx = idx[idx != 0]  
        return (idx, sum_img[idx] / counts[idx])
        
    cyt_msk = display.data[idx]["cyt_msk"]
    ncl_msk = display.data[idx]["ncl_msk"]
    C2_lbl  = display.data[idx]["C2_lbl"]
    C3_lbl  = display.data[idx]["C3_lbl"]
    
    t0 = time.time()
    print("get_edt() : ", end="", flush=False)
    
    cyt_edt = get_edt(cyt_msk > 0)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    t0 = time.time()
    print("get_obj_int() : ", end="", flush=False)
    
    val_C3  = get_obj_int(C2_lbl, (C3_lbl > 0))
    val_edt = get_obj_int(C2_lbl, cyt_edt)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Plot 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))   
    ax.scatter(val_C3[1], val_edt[1])
    ax.set_ylabel("edt")
    ax.set_xlabel("C3")
    
#%%    
        
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(cyt_msk)
    # viewer.add_image(cyt_edt)