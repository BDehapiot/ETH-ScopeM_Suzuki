#%% Imports -------------------------------------------------------------------

import time
import shutil
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# functions
from functions import format_stack, prepare_stack

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# skimage
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import (
    ball, h_maxima, remove_small_objects, remove_small_holes, binary_dilation
    )

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

#%% Comments ------------------------------------------------------------------

'''
Current:
- improve check display and rendering speed
'''

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
    "predict"    : 0,
    "process"    : 0,
    }

# Parameters
rf = 0.5
cyt_thresh = 0.05
ncl_thresh = 0.10
C1b_thresh = 0.25
C2b_thresh = 0.50
C3b_thresh = 0.33
model_name = "model_512_normal_2000-160_3"

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
htk_paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

def save(arr, path, voxsize):
    if arr.ndim == 3: axes = "ZYX"
    if arr.ndim == 4: axes = "ZCYX"
    resolution = (1 / voxsize, 1 / voxsize)
    metadata = {"axes" : axes, "spacing" : voxsize, "unit" : "um"}
    tifffile.imwrite(
        path, arr, imagej=True, resolution=resolution, metadata=metadata)

def get_masks(htk, prd):
    
    # Process
    prd_prp = norm_pct(gaussian(prd, sigma=2))
    cyt_prp = norm_pct(gaussian(htk[:, 0, ...], sigma=1))
    ncl_prp = norm_pct(gaussian(htk[:, 3, ...], sigma=2))
    cyt_prp *= prd_prp
    ncl_prp *= prd_prp
    cyt_msk = cyt_prp > cyt_thresh
    ncl_msk = ncl_prp > ncl_thresh
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

def get_blobs(arr, mask=None, sigma0=0.5, sigma1=5, thresh=0.5, out_prp=False):
    
    # preprocess
    prp = gaussian(arr, sigma=sigma0)
    prp -= gaussian(prp, sigma=sigma1)
    prp = norm_pct(prp)
    if mask is not None:
        prp[mask == 0] = 0
    msk = prp > 0.75
    
    # watershed
    hmax = h_maxima(prp, 0.1, footprint=ball(1))
    hmax = (hmax * 255).astype("uint8")
    hmax = label(hmax)
    lbl = watershed(-prp, markers=hmax, mask=msk)
    
    if out_prp:
        return lbl, prp
    else:
        return lbl
    
def get_outlines(arr):
    arr = arr > 0
    out = []
    for img in arr:
        out.append(binary_dilation(arr) ^ arr)
    return np.stack(out)

#%% Function : preprocess() ---------------------------------------------------

def preprocess(path, rf=0.5):
        
    # Format stack
    htk, voxsize = format_stack(path, rf=rf)
           
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
    htk_path = dir_path / (path.stem + f"_{voxsize}_htk.tif")
    save(htk, htk_path, voxsize)
    
#%% Function : predict() ------------------------------------------------------    

def predict(path):
            
    # Load data
    dir_path = Path(data_path / path.stem)
    htk_path = list(dir_path.glob("*_htk.tif"))[0]
    voxsize = float(str(htk_path.stem).split("_")[1])
    htk = io.imread(htk_path)
    htk = np.moveaxis(htk, -1, 1)
    
    # Predict
    prp = prepare_stack(htk)
    unet = UNet(load_name=model_name)
    prd = unet.predict(prp, verbose=0)

    # Save       
    prd_path = str(htk_path).replace("htk", "prd")
    save(prd, prd_path, voxsize)

#%% Function : process() ------------------------------------------------------

def process(
        paths, 
        cyt_thresh=0.05, 
        ncl_thresh=0.20,
        C1b_thresh=0.25,
        C2b_thresh=0.50,
        C3b_thresh=0.33,
        ):

    def _process(i, htk, prd):
        
        # Get masks
        cyt_msk, ncl_msk = get_masks(htk, prd)
        
        # Get blobs
        C1b_lbl = get_blobs(
            htk[:, 0, ...], mask=cyt_msk, 
            sigma0=0.5, sigma1=5, thresh=C1b_thresh
            )
        C2b_lbl = get_blobs(
            htk[:, 1, ...], mask=cyt_msk, 
            sigma0=0.5, sigma1=5, thresh=C2b_thresh
            )
        C3b_lbl = get_blobs(
            htk[:, 2, ...], mask=cyt_msk, 
            sigma0=0.5, sigma1=5, thresh=C3b_thresh
            )
                
        # Save
        dir_path = Path(data_path / paths[i].stem)
        htk_path = list(dir_path.glob("*_htk.tif"))[0]
        voxsize = float(str(htk_path.stem).split("_")[1])
        cyt_msk_path = str(htk_path).replace("htk", "cyt_msk")
        ncl_msk_path = str(htk_path).replace("htk", "ncl_msk")
        C1b_lbl_path = str(htk_path).replace("htk", "C1b_lbl")
        C2b_lbl_path = str(htk_path).replace("htk", "C2b_lbl")
        C3b_lbl_path = str(htk_path).replace("htk", "C3b_lbl")
        save((cyt_msk * 255).astype("uint8"), cyt_msk_path, voxsize)
        save((ncl_msk * 255).astype("uint8"), ncl_msk_path, voxsize)
        save(C1b_lbl.astype("uint16"), C1b_lbl_path, voxsize)
        save(C2b_lbl.astype("uint16"), C2b_lbl_path, voxsize)
        save(C3b_lbl.astype("uint16"), C3b_lbl_path, voxsize)
        
    # Load data
    htks, prds = [], []
    for path in paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_htk.tif"))[0]
        prd_path = list(dir_path.glob("*_prd.tif"))[0]
        htks.append(np.moveaxis(io.imread(stk_path), -1, 1))
        prds.append(io.imread(prd_path))
    
    # Process
    Parallel(n_jobs=-1)(
        delayed(_process)(i, htk, prd) 
        for i, (htk, prd) in enumerate(zip(htks, prds))
        )    

#%% Class : Display() ---------------------------------------------------------

class Display:
    
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.init_data()
        self.init_viewer()

    def init_data(self):
        
        # Load
        path = self.paths[self.idx]
        dir_path = data_path / path.stem
        htk_path = list(dir_path.glob("*_htk.tif"))[0]
        prd_path = str(htk_path).replace("htk", "prd")
        cyt_msk_path = str(htk_path).replace("htk", "cyt_msk")
        ncl_msk_path = str(htk_path).replace("htk", "ncl_msk")
        C1b_lbl_path = str(htk_path).replace("htk", "C1b_lbl")
        C2b_lbl_path = str(htk_path).replace("htk", "C2b_lbl")
        C3b_lbl_path = str(htk_path).replace("htk", "C3b_lbl")
        
        self.htk = io.imread(htk_path)
        self.C1 = self.htk[..., 0]
        self.C2 = self.htk[..., 1]
        self.C3 = self.htk[..., 2]
        self.C4 = self.htk[..., 3]
        self.prd = io.imread(prd_path)
        self.cyt_msk = io.imread(cyt_msk_path)
        self.ncl_msk = io.imread(ncl_msk_path)
        self.C1b_msk = io.imread(C1b_lbl_path) > 0
        self.C2b_msk = io.imread(C2b_lbl_path) > 0
        self.C3b_msk = io.imread(C3b_lbl_path) > 0
        
        # Get outlines (slow !!!)
        self.C1b_out = get_outlines(self.C1b_msk)
        self.C2b_out = get_outlines(self.C2b_msk)
        self.C3b_out = get_outlines(self.C3b_msk)          

    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Raws
        
        self.viewer.add_image(
            self.C1, name="C1", colormap="bop orange", visible=0,
            blending="additive", 
            gamma=0.75,
            )
        self.viewer.add_image(
            self.C2, name="C2", colormap="bop blue", visible=0,
            blending="additive", 
            )
        self.viewer.add_image(
            self.C3, name="C3", colormap="bop purple", visible=0,
            blending="additive", 
            )
        self.viewer.add_image(
            self.C4, name="C4", colormap="blue", visible=0,
            blending="additive", 
            )
        
        # Predictions
        
        self.viewer.add_image(
            self.prd, name="prd", colormap="turbo", visible=0,
            rendering="attenuated_mip", attenuation=0.5, opacity=0.25,
            )
        
        # Masks
        
        self.viewer.add_image(
            self.cyt_msk, name="cyt_msk", colormap="gray", visible=1,
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.ncl_msk, name="ncl_msk", colormap="blue", visible=1,
            blending="translucent_no_depth", opacity=0.2,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        # Blobs
        
        self.viewer.add_image(
            self.C1b_msk, name="C1b_msk", colormap="bop orange", visible=1,
            blending="additive", opacity=0.75, 
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.C2b_msk, name="C2b_msk", colormap="bop blue", visible=1,
            blending="additive", opacity=0.75, 
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.C3b_msk, name="C3b_msk", colormap="bop purple", visible=1,
            blending="additive", opacity=0.75, 
            rendering="attenuated_mip", attenuation=0.5, 
            )
        
        # Outlines
        
        self.viewer.add_image(
            self.C1b_out, name="C1b_out", colormap="gray", visible=0,
            blending="additive", opacity=0.5, 
            )
        self.viewer.add_image(
            self.C2b_out, name="C2b_out", colormap="gray", visible=0,
            blending="additive", opacity=0.5, 
            )
        self.viewer.add_image(
            self.C3b_out, name="C3b_out", colormap="gray", visible=0,
            blending="additive", opacity=0.5,
            )

        
        # 3D display
        self.viewer.dims.ndisplay = 3
        
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
        self.info_path.setText(
            f"{self.paths[self.idx].name}"
            )
        self.info_shortcuts = QLabel()
        self.info_shortcuts.setFont(QFont("Consolas"))
        self.info_shortcuts.setText(
            "prev/next stack  : page down/up \n"
            )
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.htk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addWidget(self.chk_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_path)
        self.layout.addWidget(self.info_shortcuts)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_hstack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_hstack()
        
    # Methods
        
    def update_layers(self):
        self.viewer.layers["C1"].data = self.C1
        self.viewer.layers["C2"].data = self.C2
        self.viewer.layers["C3"].data = self.C3
        self.viewer.layers["C4"].data = self.C4
        self.viewer.layers["prd"].data = self.prd
        self.viewer.layers["cyt_msk"].data = self.cyt_msk
        self.viewer.layers["ncl_msk"].data = self.ncl_msk
        self.viewer.layers["C1b_msk"].data = self.C1b_msk
        self.viewer.layers["C2b_msk"].data = self.C2b_msk
        self.viewer.layers["C3b_msk"].data = self.C3b_msk
        
    def update_text(self):
        self.info_path.setText(f"{self.paths[self.idx].name}")
        
    def show_masks(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if "msk" not in name:
                self.viewer.layers[name].visible = 0
            else:
                self.viewer.layers[name].visible = 1
    
    def show_predictions(self):
        self.viewer.dims.ndisplay = 3
        for name in self.viewer.layers:
            name = str(name)
            if name not in ["C1", "C4", "prd"]:
                self.viewer.layers[name].visible = 0
            else:
                self.viewer.layers[name].visible = 1
 
    def show_chk(self, tag="C1"):
        self.viewer.dims.ndisplay = 2
        for name in self.viewer.layers:
            name = str(name)
            if tag not in name:
                self.viewer.layers[name].visible = 0
            else:
                self.viewer.layers[name].visible = 1
        
    # Shortcuts
        
    def next_hstack(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.init_data()
            self.update_layers()
            self.update_text()
            
    def prev_hstack(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_layers()
            self.update_text()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Preprocess --------------------------------------------------------------

    # Paths
    paths = []
    for path in htk_paths:
        dir_path = Path(data_path / path.stem)
        htk_path = list(dir_path.glob("*_htk.tif"))
        if not htk_path or not htk_path[0].exists() or overwrite["preprocess"]:
            paths.append(path)
    
    print("preprocess : ", end="", flush=True)
    t0 = time.time()
    
    # Execute
    if paths:
        Parallel(n_jobs=-1)(
            delayed(preprocess)(path, rf=rf) 
            for path in paths
            )  
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Predict -----------------------------------------------------------------
    
    # Paths
    paths = []
    for path in htk_paths:
        dir_path = Path(data_path / path.stem)
        prd_path = list(dir_path.glob("*_prd.tif"))
        if not prd_path or not prd_path[0].exists() or overwrite["predict"]:
            paths.append(path)
    
    print("predict    : ", end="", flush=True)
    t0 = time.time()
    
    if paths:
        for path in paths:
            predict(path)   
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Process -----------------------------------------------------------------

    # Paths
    paths = []
    for path in htk_paths:
        dir_path = Path(data_path / path.stem)
        cyt_msk_path = list(dir_path.glob("*_cyt_msk.tif"))
        if not cyt_msk_path or not cyt_msk_path[0].exists() or overwrite["process"]:
            paths.append(path)

    print("process    : ", end="", flush=True)
    t0 = time.time()
    
    if paths:
        process(
            paths, 
            cyt_thresh=cyt_thresh,
            ncl_thresh=ncl_thresh,
            )
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display -----------------------------------------------------------------
    
    Display(htk_paths)
