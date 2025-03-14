#%% Imports -------------------------------------------------------------------

import nd2
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
from skimage.transform import rescale
from skimage.filters.rank import median
from skimage.filters import threshold_otsu
from skimage.morphology import ball, h_maxima, remove_small_objects

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QGroupBox, QVBoxLayout, 
    )

#%% Comments ------------------------------------------------------------------

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
    "process"    : 1,
    }

# Parameters
rf = 0.5
cyt_thresh = 0.05
ncl_thresh = 0.20

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:\local_Suzuki\data")
stk_paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

def save(stk, path, voxsize):
    if stk.ndim == 3: axes = "ZYX"
    if stk.ndim == 4: axes = "ZCYX"
    resolution = (1 / voxsize, 1 / voxsize)
    metadata = {"axes" : axes, "spacing" : voxsize, "unit" : "um"}
    tifffile.imwrite(
        path, stk, imagej=True, resolution=resolution, metadata=metadata)

#%% Function : preprocess() ---------------------------------------------------

def preprocess(path, rf=0.5):
        
    # Format stack
    stk, voxsize = format_stack(path, rf=rf)
           
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
    stk_path = dir_path / (path.stem + f"_{voxsize}_stk.tif")
    save(stk, stk_path, voxsize)
    
#%% Function : predict() ------------------------------------------------------    

def predict(path):
            
    # Load data
    dir_path = Path(data_path / path.stem)
    stk_path = list(dir_path.glob("*_stk.tif"))[0]
    voxsize = float(str(stk_path.stem).split("_")[1])
    stk = io.imread(stk_path)
    stk = np.moveaxis(stk, -1, 1)
    
    # Predict
    prp = prepare_stack(stk)
    unet = UNet(load_name="model_512_normal_1000-160_2")
    prd = unet.predict(prp, verbose=0)

    # Save       
    prd_path = str(stk_path).replace("stk", "prd")
    save(prd, prd_path, voxsize)

#%% Function : process() ------------------------------------------------------

def process(paths, cyt_thresh=0.05, ncl_thresh=0.20):
    
    global stk, cyts, ncls, prds
    
    def _process(i, cyt, ncl, prd):
        
        # Process
        prd_prp = norm_pct(gaussian(prd, sigma=2))
        cyt_prp = norm_pct(gaussian(cyt, sigma=1))
        ncl_prp = norm_pct(gaussian(ncl, sigma=1))
        cyt_prp *= prd_prp
        ncl_prp *= prd_prp
        cyt_msk = cyt_prp > cyt_thresh
        ncl_msk = ncl_prp > ncl_thresh
        cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
        ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
        cyt_msk[ncl_msk] = True
        
        # Save
        dir_path = Path(data_path / paths[i].stem)
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        voxsize = float(str(stk_path.stem).split("_")[1])
        cyt_msk_path = str(stk_path).replace("stk", "cyt_msk")
        ncl_msk_path = str(stk_path).replace("stk", "ncl_msk")
        save((cyt_msk * 255).astype("uint8"), cyt_msk_path, voxsize)
        save((ncl_msk * 255).astype("uint8"), ncl_msk_path, voxsize)
        
    # Load data
    cyts, ncls, prds = [], [], []
    for path in paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        prd_path = list(dir_path.glob("*_prd.tif"))[0]
        stk = np.moveaxis(io.imread(stk_path), -1, 1)
        prds.append(io.imread(prd_path))
        cyts.append(stk[:, 0, ...])
        ncls.append(stk[:, 3, ...])
    
    # Process
    Parallel(n_jobs=-1)(
        delayed(_process)(i, cyt, ncl, prd) 
        for i, (cyt, ncl, prd) in enumerate(zip(cyts, ncls, prds))
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
        stk_path = list(dir_path.glob("*_stk.tif"))[0]
        cyt_msk_path = str(stk_path).replace("stk", "cyt_msk")
        ncl_msk_path = str(stk_path).replace("stk", "ncl_msk")
        prd_path = str(stk_path).replace("stk", "prd")
        
        self.stk = io.imread(stk_path)
        self.C1 = self.stk[..., 0]
        self.C2 = self.stk[..., 1]
        self.C3 = self.stk[..., 2]
        self.C4 = self.stk[..., 3]
        self.cyt_msk = io.imread(cyt_msk_path)
        self.ncl_msk = io.imread(ncl_msk_path)
        self.prd = io.imread(prd_path)
        
    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            self.C1, name="NRP2", colormap="bop orange", visible=True,
            blending="additive", 
            gamma=0.75,
            )
        self.viewer.add_image(
            self.C2, name="virus", colormap="green", visible=False,
            blending="additive", 
            )
        self.viewer.add_image(
            self.C3, name="EEA1", colormap="magenta", visible=False,
            blending="additive", 
            )
        self.viewer.add_image(
            self.C4, name="nuclei", colormap="bop blue", visible=True,
            blending="additive", 
            )
        self.viewer.add_image(
            self.prd, name="prd", colormap="gray", visible=False,
            rendering="attenuated_mip", attenuation=0.5, opacity=0.25,
            )
        self.viewer.add_image(
            self.cyt_msk, name="cyt_msk", colormap="bop orange", visible=True,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        self.viewer.add_image(
            self.ncl_msk, name="ncl_msk", colormap="bop blue", visible=True,
            rendering="attenuated_mip", attenuation=0.5, 
            )

        
        # 3D display
        self.viewer.dims.ndisplay = 3
        
        # Create "select stack" menu
        self.stk_group_box = QGroupBox("Select stack")
        stk_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        stk_group_layout.addWidget(self.btn_next_image)
        stk_group_layout.addWidget(self.btn_prev_image)
        self.stk_group_box.setLayout(stk_group_layout)
        self.btn_next_image.clicked.connect(self.next_stack)
        self.btn_prev_image.clicked.connect(self.prev_stack)
        
        # Create texts
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_image.setText(
            f"{self.paths[self.idx].name}"
            )
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stk_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_stack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_stack()
        
    def update_stack(self):
        self.viewer.layers["NRP2"].data = self.C1
        self.viewer.layers["virus"].data = self.C2
        self.viewer.layers["EEA1"].data = self.C3
        self.viewer.layers["nuclei"].data = self.C4
        self.viewer.layers["cyt_msk"].data = self.cyt_msk
        self.viewer.layers["ncl_msk"].data = self.ncl_msk
        self.viewer.layers["prd"].data = self.prd
        
    def update_text(self):
        self.info_image.setText(f"{self.paths[self.idx].name}")
        
    def next_stack(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.init_data()
            self.update_stack()
            self.update_text()
            
    def prev_stack(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_stack()
            self.update_text()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Preprocess --------------------------------------------------------------

    # Paths
    paths = []
    for path in stk_paths:
        dir_path = Path(data_path / path.stem)
        stk_path = list(dir_path.glob("*_stk.tif"))
        if not stk_path or not stk_path[0].exists() or overwrite["preprocess"]:
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
    for path in stk_paths:
        dir_path = Path(data_path / path.stem)
        prd_path = list(dir_path.glob("*_prd.tif"))
        if not prd_path or not prd_path[0].exists() or overwrite["predict"]:
            paths.append(path)
    
    print("predict : ", end="", flush=True)
    t0 = time.time()
    
    if paths:
        for path in paths:
            predict(path)   
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Process -----------------------------------------------------------------

    # Paths
    paths = []
    for path in stk_paths:
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
    
    Display(stk_paths)

#%%
    
    # # Fetch
    # idx = 0
    # path = stk_paths[idx]
    # dir_path = Path(data_path / path.stem)
    # stk_path = list(dir_path.glob("*_stk.tif"))[0]
    # cyt_msk_path = str(stk_path).replace("stk", "cyt_msk")
    # stk = np.moveaxis(io.imread(stk_path), -1, 1)
    # cyt_msk = io.imread(cyt_msk_path)
    # C1 = stk[:, 0, ...]
    # C2 = stk[:, 1, ...]
    
    # # Process
    # C2_prp = C2.copy()
    # C2_prp = norm_pct(C2_prp)
    # C2_prp[cyt_msk == 0] = 0
    # C2_hmax = h_maxima(C2_prp, 0.1, footprint=ball(1))
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.dims.ndisplay = 3
    
    # # viewer.add_image(
    # #     C1, name="C1", colormap="bop orange", visible=1,
    # #     blending="additive", gamma=0.5,
    # #     )
    # viewer.add_image(
    #     C2_prp, name="C2_prp", colormap="magenta", visible=1,
    #     blending="additive", gamma=0.5,
    #     )
    
    # # viewer.add_image(
    # #     cyt_msk, name="cyt_msk", colormap="bop orange", visible=1,
    # #     rendering="attenuated_mip", attenuation=0.5, 
    # #     )
    # viewer.add_image(
    #     C2_hmax, name="C2_hmax", colormap="gray", visible=1,
    #     rendering="attenuated_mip", attenuation=0.5, 
    #     )

#%%

    # # Fetch
    # idx = 2
    # cyt = cyts[idx]
    # ncl = ncls[idx]
    # prd = prds[idx]
    
    # # 
    # prd_prp = norm_pct(gaussian(prd, sigma=1))
    # cyt_prp = norm_pct(gaussian(cyt, sigma=1))
    # ncl_prp = norm_pct(gaussian(ncl, sigma=1))
    # cyt_prp *= prd_prp
    # ncl_prp *= prd_prp
    # cyt_msk = cyt_prp > 0.05
    # ncl_msk = ncl_prp > 0.20
    # cyt_msk = remove_small_objects(cyt_msk, min_size=1e4)
    # ncl_msk = remove_small_objects(ncl_msk, min_size=1e4)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.dims.ndisplay = 3
    
    # viewer.add_image(
    #     cyt, name="cyt", colormap="bop orange", visible=0,
    #     blending="additive", 
    #     )
    # viewer.add_image(
    #     ncl, name="ncl", colormap="bop blue", visible=0,
    #     blending="additive", 
    #     )
    # viewer.add_image(
    #     prd, name="prd", colormap="gray", visible=0,
    #     rendering="attenuated_mip", attenuation=0.5, opacity=0.5, 
    #     )
    
    # viewer.add_image(
    #     prd_prp, name="prd", colormap="gray", visible=1,
    #     rendering="attenuated_mip", attenuation=0.5, opacity=0.5, 
    #     )
    # viewer.add_image(
    #     cyt_prp, name="cyt_prp", colormap="bop orange", visible=1,
    #     blending="additive", 
    #     )
    # viewer.add_image(
    #     cyt_msk, name="cyt_msk", colormap="bop orange", visible=1,
    #     rendering="attenuated_mip", attenuation=0.5, 
    #     )
    # viewer.add_image(
    #     ncl_prp, name="ncl_prp", colormap="bop blue", visible=1,
    #     blending="additive", 
    #     )
    # viewer.add_image(
    #     ncl_msk, name="ncl_msk", colormap="bop blue", visible=1,
    #     rendering="attenuated_mip", attenuation=0.5, 
    #     )